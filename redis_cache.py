import os
import json
import numpy as np
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.index_definition import IndexDefinition, IndexType
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import sys

logger = logging.getLogger(__name__)

load_dotenv()

# --- CONFIGURATION ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL = 3600  
INDEX_NAME = "rag_cache_idx"

# Semantic Threshold: Lower distance = more similar. 
# 0.15 means the query must be roughly 85%+ mathematically similar to trigger a cache hit.
SIMILARITY_THRESHOLD = 0.30

# Load a fast, lightweight local embedding model to vectorize the queries
logger.info("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_DIMENSION = 384 # MiniLM creates vectors with 384 dimensions

def init_redis():
    """Connect to Redis and create the Vector Index if it doesn't exist."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        r.ping()
        
        # Check if the vector index already exists
        try:
            r.ft(INDEX_NAME).info()
            logger.info("Redis Semantic Cache connected (Index exists).")
        except redis.exceptions.ResponseError:
            # Create the vector index schema
            schema = (
                TextField("original_query"),
                TextField("cached_result"),
                VectorField("query_vector", 
                    "FLAT", {
                        "TYPE": "FLOAT32", 
                        "DIM": VECTOR_DIMENSION, 
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            )
            definition = IndexDefinition(prefix=["rag_cache:"], index_type=IndexType.HASH)
            r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
            logger.info("Redis Semantic Cache connected (New Index created).")
            
        return r
    except redis.ConnectionError:
        logger.info("Warning: Could not connect to Redis. Caching is disabled.")
        return None

# Global Redis Client
redis_client = init_redis()

def get_cached_results(query: str, n_result: int):
    """Embeds the query and searches Redis for semantically similar past queries."""
    if not redis_client:
        return None
        
    try:
        # 1. Convert the text query into a mathematical vector
        query_vector = embedder.encode(query).astype(np.float32).tobytes()
        
        # 2. Build a K-Nearest Neighbor (KNN) Search Query
        # We are asking Redis to find the 1 closest vector to our query vector
        search_query = (
            Query(f"*=>[KNN 1 @query_vector $vec AS vector_score]")
            .sort_by("vector_score")
            .return_fields("vector_score", "original_query", "cached_result")
            .dialect(2)
        )
        
        # 3. Execute the search
        results = redis_client.ft(INDEX_NAME).search(
            search_query, 
            query_params={"vec": query_vector}
        )
        
        # 4. Evaluate the similarity
        if results.docs:
            closest_match = results.docs[0]
            distance_score = float(closest_match.vector_score)
            
            # If the mathematical distance is below our threshold, it's a hit!
            if distance_score < SIMILARITY_THRESHOLD:
                logger.info(f"SEMANTIC CACHE HIT: '{query}' matched past query '{closest_match.original_query}' (Distance: {distance_score:.4f})")
                return json.loads(closest_match.cached_result)
            else:
                logger.info(f"CACHE MISS: Closest match was '{closest_match.original_query}' but distance was too high ({distance_score:.4f})")
                
    except Exception as e:
        logger.info(f"Redis search error: {e}")
        
    return None

def set_cached_results(query: str, n_result: int, results: dict):
    """Embeds the query and saves the vector, original text, and results as a Redis Hash."""
    if not redis_client:
        return
        
    try:
        # Create a unique ID for this cache entry
        import uuid
        cache_key = f"rag_cache:{uuid.uuid4().hex}"
        
        # Convert text to vector
        query_vector = embedder.encode(query).astype(np.float32).tobytes()
        
        # Store data in a Redis Hash
        redis_client.hset(cache_key, mapping={
            "original_query": query,
            "query_vector": query_vector,
            "cached_result": json.dumps(results)
        })
        
        # Set Time-to-Live (TTL)
        redis_client.expire(cache_key, CACHE_TTL)
        
    except Exception as e:
        logger.info(f"Redis write error: {e}")

def clear_cache():
    """Flushes the database."""
    if redis_client:
        try:
            redis_client.flushdb()
            logger.info("Redis cache cleared.")
        except Exception as e:
            logger.info(f"Redis flush error: {e}")