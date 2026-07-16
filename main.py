import os
import logging
import sys 
from fastmcp import FastMCP
import chromadb
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from redis_cache import get_cached_results,set_cached_results,clear_cache

logging.basicConfig(
  level=logging.INFO,
  format="%(message)s",
  stream=sys.stderr
)

logger = logging.getLogger(__name__)


load_dotenv()

PERSISTENT_CLIENT = "./chroma_db"
COLLECTION_NAME = "rag_mcp"
DATA_DIR = "./data"
LLAMA_API = os.getenv("LLAMA_CLOUD_API_KEY", "")

mcp = FastMCP("RAG Server")


def init_chroma():
  client = chromadb.PersistentClient(path=PERSISTENT_CLIENT)

  collection = client.get_or_create_collection(name=COLLECTION_NAME)

  return client, collection


@mcp.tool(timeout=600)
def ingest_data():
  """
  Initialize chromadb and collections so that the user can query them later.
  """

  client, collection = init_chroma()
  client.delete_collection(name=COLLECTION_NAME)
  new_collection = client.get_or_create_collection(name=COLLECTION_NAME)


  parser = LlamaParse(api_key=LLAMA_API, result_type="text")

  file_extractor = {'.pdf' : parser}

  documents = SimpleDirectoryReader(DATA_DIR, file_extractor=file_extractor).load_data()

  for doc in documents:
    new_collection.add(
      documents=[doc.text],
      metadatas=[doc.metadata],
      ids=[doc.doc_id]
    )
  clear_cache()

  final_count = new_collection.count()
  return f"Final count: {final_count}"


@mcp.tool(timeout=600)
def query_data(query:str , n_result:int):
   """
   Query the vector database for documents similar to the query.
   """
   cached_data = get_cached_results(query, n_result)
   if cached_data:
     return cached_data
   
   logger.info(f"CACHE MISS: Querying ChromaDB for '{query}' ......")

   client , collection = init_chroma()

   collection = client.get_collection(name=COLLECTION_NAME)

   results = collection.query(
      query_texts=[query],
      n_results= n_result,
      include=["documents", "metadatas", "distances"]
   )

   if results == None or len(results) == 0:
      return "No results found"
   
   set_cached_results(query,n_result,results)
   return results


@mcp.tool(timeout=600)
def get_db_status():
  """
  Get the status of the vector database.
  """

  chroma_client, chroma_collection = init_chroma()
  count = chroma_collection.count()
  return f"Database status: {count} documents ingested."

if __name__ == "__main__":
  init_chroma()
  mcp.run("stdio")