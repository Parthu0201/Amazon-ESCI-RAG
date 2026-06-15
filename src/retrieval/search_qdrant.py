import logging
import os
import sys
import time
import textwrap

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_PATH = "vectorstore/qdrant_db"
COLLECTION_NAME = "esci_products"

# Lazy initialization for reusability
_model = None
_client = None


def get_model() -> Any:
    """
    Load and return the SentenceTransformer embedding model.
    Implements a singleton pattern to avoid reloading.
    """
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding
            logger.info(f"Loading fastembed model: {MODEL_NAME}")
            _model = TextEmbedding(model_name=MODEL_NAME)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model {MODEL_NAME}: {e}")
            raise RuntimeError(f"Model loading error: {e}")
    return _model


def get_qdrant_client() -> QdrantClient:
    """
    Initialize and return the local Qdrant client.
    Implements a singleton pattern to avoid multiple connections.
    """
    global _client
    if _client is None:
        logger.info(f"Connecting to Qdrant at path: {QDRANT_PATH}")
        try:
            _client = QdrantClient(path=QDRANT_PATH)
            logger.info("Connected to Qdrant successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Qdrant connection error: {e}")
    return _client


def search_products(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search for similar products in the Qdrant vector database.

    Args:
        query (str): The user's search query.
        top_k (int, optional): Number of top results to retrieve. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the retrieved products
            and their metadata, along with similarity scores.
            
    Raises:
        ValueError: If query is empty or collection is missing.
        ConnectionError: If Qdrant connection fails.
        RuntimeError: If embedding generation or search fails.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not query or not str(query).strip():
        logger.error("Empty query received.")
        raise ValueError("Query string cannot be empty.")

    query = query.lower().strip()
    logger.info(f"Query received: '{query}'")

    # Ensure connection and collection exist
    client = get_qdrant_client()
    try:
        collections_response = client.get_collections()
        collections = [col.name for col in collections_response.collections]
        
        if COLLECTION_NAME not in collections:
            logger.error(f"Collection '{COLLECTION_NAME}' does not exist.")
            raise ValueError(f"Collection '{COLLECTION_NAME}' missing. Please build the index first.")
    except UnexpectedResponse as e:
         logger.error(f"Qdrant UnexpectedResponse during collection check: {e}")
         raise ConnectionError(f"Qdrant error: {e}")
    except Exception as e:
         logger.error(f"Error checking collections: {e}")
         raise ConnectionError(f"Qdrant error: {e}")

    # Generate embeddings
    model = get_model()
    try:
        logger.info("Generating query embedding...")
        query_embedding_generator = model.embed([query])
        query_embedding = list(query_embedding_generator)[0].tolist()
    except Exception as e:
        logger.error(f"Failed to generate embedding for query: {e}")
        raise RuntimeError(f"Embedding generation error: {e}")

    # Search in Qdrant
    try:
        logger.info(f"Searching Qdrant collection '{COLLECTION_NAME}' for top {top_k} results...")
        start_time = time.time()
        
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        search_results = response.points
        
        end_time = time.time()
        logger.info(f"Retrieval completed in {end_time - start_time:.3f} seconds")
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        raise RuntimeError(f"Qdrant search error: {e}")

    logger.info(f"Number of results returned: {len(search_results)}")

    # Format results
    formatted_results = []
    for hit in search_results:
        payload = hit.payload or {}
        similarity_score = round(float(hit.score), 4)
        vector_score = float(hit.score)
        
        formatted_results.append({
            "product_id": payload.get("product_id", "N/A"),
            "product_title": payload.get("product_title", "N/A"),
            "product_brand": payload.get("product_brand", "N/A"),
            "product_bullet_point": payload.get("product_bullet_point", "N/A"),
            "product_description": payload.get("product_description", "N/A"),
            "split": payload.get("split", "N/A"),
            "similarity_score": similarity_score,
            "vector_score": vector_score
        })

    return formatted_results


def display_results(query: str, results: List[Dict[str, Any]]):
    """
    Formats and prints the search results cleanly to the console.
    
    Args:
        query (str): The search query.
        results (List[Dict[str, Any]]): The retrieved search results.
    """
    print("\n" + "=" * 70)
    print(f"🔍 SEARCH RESULTS FOR: '{query.upper()}'")
    print("=" * 70 + "\n")

    if not results:
        print("   No results found.")
        print("=" * 70 + "\n")
        return

    for rank, res in enumerate(results, start=1):
        print(f" ⭐ Rank #{rank} (Score: {res['similarity_score']:.4f})")
        print(f"    ID:    {res['product_id']}")
        print(f"    Brand: {res['product_brand']}")
        
        title = textwrap.fill(str(res['product_title']), width=70, subsequent_indent="           ")
        print(f"    Title: {title}")
        
        bullet_points = res.get('product_bullet_point')
        if bullet_points and str(bullet_points).strip() not in ["N/A", "None", ""]:
            # Truncate very long bullet points for display
            bp_str = str(bullet_points)
            if len(bp_str) > 250:
                bp_str = bp_str[:247] + "..."
            bp_text = textwrap.fill(bp_str, width=70, initial_indent="    • ", subsequent_indent="      ")
            print(f"\n    Highlights:\n{bp_text}")
            
        description = res.get('product_description')
        if description and str(description).strip() not in ["N/A", "None", ""]:
            # Truncate very long descriptions
            desc_str = str(description)
            if len(desc_str) > 250:
                desc_str = desc_str[:247] + "..."
            desc_text = textwrap.fill(desc_str, width=70, initial_indent="    Desc:  ", subsequent_indent="           ")
            print(f"\n{desc_text}")
            
        print("\n" + "-" * 70 + "\n")
        
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Standard test queries
    test_queries = [
        "wireless mouse",
        "iphone charger",
        "bluetooth speaker",
        "gaming keyboard"
    ]
    
    print("Initializing ESCI Product Search System...\n", flush=True)
    
    try:
        # Pre-load model and connect to Qdrant to avoid delays during search
        print("STEP 1: Starting", flush=True)
        print("NOTE: Please wait! The script is downloading the embedding model.", flush=True)
        print("      This may take a minute and might appear to be 'frozen'. Do not close it!", flush=True)
        get_model()
        print("STEP 2: Model Loaded", flush=True)
        get_qdrant_client()
        print("STEP 3: Qdrant Connected", flush=True)
        
        # Run automated test queries
        for q in test_queries:
            try:
                print(f"Testing query: {q}", flush=True)
                results = search_products(query=q, top_k=2)
                display_results(q, results)
            except Exception as e:
                logger.error(f"Error during automated test query '{q}': {e}")
        
        # Start interactive CLI session
        print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                user_query = input("\nEnter a search query: ").strip()
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("Exiting search system. Goodbye!")
                    break
                if not user_query:
                    continue
                    
                results = search_products(query=user_query, top_k=5)
                display_results(user_query, results)
            except KeyboardInterrupt:
                print("\nExiting search system. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                
        # Cleanly close Qdrant connection to avoid __del__ shutdown error
        if _client is not None:
            _client.close()
                
    except Exception as e:
        print(f"\nFATAL ERROR: System initialization failed: {e}", flush=True)
        logger.critical(f"System initialization failed: {e}")
        sys.exit(1)











# import logging
# import os
# import sys
# import time

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from typing import List, Dict, Any

# from qdrant_client import QdrantClient
# from qdrant_client.http.exceptions import UnexpectedResponse

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
# logger = logging.getLogger(__name__)

# # Constants
# MODEL_NAME = "BAAI/bge-small-en-v1.5"
# QDRANT_PATH = "vectorstore/qdrant_db"
# COLLECTION_NAME = "esci_products"

# # Lazy initialization for reusability
# _model = None
# _client = None


# def get_model() -> Any:
#     """
#     Load and return the SentenceTransformer embedding model.
#     Implements a singleton pattern to avoid reloading.
#     """
#     global _model
#     if _model is None:
#         try:
#             from fastembed import TextEmbedding
#             logger.info(f"Loading fastembed model: {MODEL_NAME}")
#             _model = TextEmbedding(model_name=MODEL_NAME)
#             logger.info("Model loaded successfully.")
#         except Exception as e:
#             logger.error(f"Failed to load model {MODEL_NAME}: {e}")
#             raise RuntimeError(f"Model loading error: {e}")
#     return _model


# def get_qdrant_client() -> QdrantClient:
#     """
#     Initialize and return the local Qdrant client.
#     Implements a singleton pattern to avoid multiple connections.
#     """
#     global _client
#     if _client is None:
#         logger.info(f"Connecting to Qdrant at path: {QDRANT_PATH}")
#         try:
#             _client = QdrantClient(path=QDRANT_PATH)
#             logger.info("Connected to Qdrant successfully.")
#         except Exception as e:
#             logger.error(f"Failed to connect to Qdrant: {e}")
#             raise ConnectionError(f"Qdrant connection error: {e}")
#     return _client


# def search_products(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
#     """
#     Search for similar products in the Qdrant vector database.

#     Args:
#         query (str): The user's search query.
#         top_k (int, optional): Number of top results to retrieve. Defaults to 10.

#     Returns:
#         List[Dict[str, Any]]: A list of dictionaries containing the retrieved products
#             and their metadata, along with similarity scores.
            
#     Raises:
#         ValueError: If query is empty or collection is missing.
#         ConnectionError: If Qdrant connection fails.
#         RuntimeError: If embedding generation or search fails.
#     """
#     if top_k <= 0:
#         raise ValueError("top_k must be greater than 0")

#     if not query or not str(query).strip():
#         logger.error("Empty query received.")
#         raise ValueError("Query string cannot be empty.")

#     query = query.lower().strip()
#     logger.info(f"Query received: '{query}'")

#     # Ensure connection and collection exist
#     client = get_qdrant_client()
#     try:
#         collections_response = client.get_collections()
#         collections = [col.name for col in collections_response.collections]
        
#         if COLLECTION_NAME not in collections:
#             logger.error(f"Collection '{COLLECTION_NAME}' does not exist.")
#             raise ValueError(f"Collection '{COLLECTION_NAME}' missing. Please build the index first.")
#     except UnexpectedResponse as e:
#          logger.error(f"Qdrant UnexpectedResponse during collection check: {e}")
#          raise ConnectionError(f"Qdrant error: {e}")
#     except Exception as e:
#          logger.error(f"Error checking collections: {e}")
#          raise ConnectionError(f"Qdrant error: {e}")

#     # Generate embeddings
#     model = get_model()
#     try:
#         logger.info("Generating query embedding...")
#         query_embedding_generator = model.embed([query])
#         query_embedding = list(query_embedding_generator)[0].tolist()
#     except Exception as e:
#         logger.error(f"Failed to generate embedding for query: {e}")
#         raise RuntimeError(f"Embedding generation error: {e}")

#     # Search in Qdrant
#     try:
#         logger.info(f"Searching Qdrant collection '{COLLECTION_NAME}' for top {top_k} results...")
#         start_time = time.time()
        
#         response = client.query_points(
#             collection_name=COLLECTION_NAME,
#             query=query_embedding,
#             limit=top_k,
#             with_payload=True
#         )
#         search_results = response.points
        
#         end_time = time.time()
#         logger.info(f"Retrieval completed in {end_time - start_time:.3f} seconds")
#     except Exception as e:
#         logger.error(f"Qdrant search failed: {e}")
#         raise RuntimeError(f"Qdrant search error: {e}")

#     logger.info(f"Number of results returned: {len(search_results)}")

#     # Format results
#     formatted_results = []
#     for hit in search_results:
#         payload = hit.payload or {}
        
#         similarity_score = round(float(hit.score), 4)
        
#         formatted_results.append({
#             "product_id": payload.get("product_id", "N/A"),
#             "product_title": payload.get("product_title", "N/A"),
#             "product_brand": payload.get("product_brand", "N/A"),
#             "product_bullet_point": payload.get("product_bullet_point", "N/A"),
#             "product_description": payload.get("product_description", "N/A"),
#             "split": payload.get("split", "N/A"),
#             "similarity_score": similarity_score
#         })

#     return formatted_results


# def display_results(query: str, results: List[Dict[str, Any]]):
#     """
#     Formats and prints the search results cleanly to the console.
    
#     Args:
#         query (str): The search query.
#         results (List[Dict[str, Any]]): The retrieved search results.
#     """
#     print("\n" + "=" * 50)
#     print(f"Query: {query}")
#     print("=" * 50 + "\n")

#     if not results:
#         print("No results found.")
#         print("=" * 50 + "\n")
#         return

#     for rank, res in enumerate(results, start=1):
#         print(f"Rank #{rank}")
#         print(f"Product ID: {res['product_id']}")
#         print(f"Title: {res['product_title']}")
#         print(f"Brand: {res['product_brand']}")
#         print(f"Score: {res['similarity_score']:.4f}\n")
        
#         bullet_points = res.get('product_bullet_point')
#         if bullet_points and str(bullet_points).strip() not in ["N/A", "None", ""]:
#             print("Bullet Points:")
#             print(f"{bullet_points}\n")
            
#         description = res.get('product_description')
#         if description and str(description).strip() not in ["N/A", "None", ""]:
#             print("Description:")
#             print(f"{description}\n")
            
#         print("-" * 50 + "\n")
        
#     print("=" * 50 + "\n")


# if __name__ == "__main__":
#     # Standard test queries
#     test_queries = [
#         "wireless mouse",
#         "iphone charger",
#         "bluetooth speaker",
#         "gaming keyboard"
#     ]
    
#     print("Initializing ESCI Product Search System...\n", flush=True)
    
#     try:
#         # Pre-load model and connect to Qdrant to avoid delays during search
#         print("STEP 1: Starting", flush=True)
#         print("NOTE: Please wait! The script is downloading the embedding model.", flush=True)
#         print("      This may take a minute and might appear to be 'frozen'. Do not close it!", flush=True)
#         get_model()
#         print("STEP 2: Model Loaded", flush=True)
#         get_qdrant_client()
#         print("STEP 3: Qdrant Connected", flush=True)
        
#         # Run automated test queries
#         for q in test_queries:
#             try:
#                 print(f"Testing query: {q}", flush=True)
#                 results = search_products(query=q, top_k=2)
#                 display_results(q, results)
#             except Exception as e:
#                 logger.error(f"Error during automated test query '{q}': {e}")
        
#         # Start interactive CLI session
#         print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.")
#         while True:
#             try:
#                 user_query = input("\nEnter a search query: ").strip()
#                 if user_query.lower() in ['quit', 'exit', 'q']:
#                     print("Exiting search system. Goodbye!")
#                     break
#                 if not user_query:
#                     continue
                    
#                 results = search_products(query=user_query, top_k=5)
#                 display_results(user_query, results)
#             except KeyboardInterrupt:
#                 print("\nExiting search system. Goodbye!")
#                 break
#             except Exception as e:
#                 logger.error(f"Error processing query: {e}")
                
#         # Cleanly close Qdrant connection to avoid __del__ shutdown error
#         if _client is not None:
#             _client.close()
                
#     except Exception as e:
#         print(f"\nFATAL ERROR: System initialization failed: {e}", flush=True)
#         logger.critical(f"System initialization failed: {e}")
#         sys.exit(1)
