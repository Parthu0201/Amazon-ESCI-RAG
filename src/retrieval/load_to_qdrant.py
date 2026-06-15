import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_embeddings_to_qdrant(
    embeddings_path: str = "vectorstore/embeddings/product_embeddings.npy",
    metadata_path: str = "vectorstore/metadata/product_metadata.parquet",
    qdrant_path: str = "vectorstore/qdrant_db",
    collection_name: str = "esci_products",
    batch_size: int = 1000
) -> None:
    """
    Loads precomputed product embeddings and metadata into a local Qdrant vector database.
    
    Args:
        embeddings_path (str): Path to the generated product embeddings numpy file.
        metadata_path (str): Path to the parquet metadata file.
        qdrant_path (str): Local path for Qdrant persistence.
        collection_name (str): Name of the Qdrant collection.
        batch_size (int): Batch size for vector upload.
    """
    logger.info("Starting process to load embeddings into Qdrant.")
    total_start_time = time.time()
    
    # Validation 1: Files exist
    emb_file = Path(embeddings_path)
    meta_file = Path(metadata_path)
    
    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    # Load data
    logger.info("Loading embeddings and metadata...")
    data_loading_start = time.time()
    embeddings = np.load(embeddings_path)
    metadata_df = pd.read_parquet(metadata_path)
    data_loading_time = time.time() - data_loading_start
    
    # Validation 2: Dimensions and empty checks
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2-dimensional. Found: {embeddings.ndim}")
        
    num_vectors, vector_dim = embeddings.shape
    
    if num_vectors <= 0:
        raise ValueError("Embeddings file is empty. num_vectors must be > 0.")
        
    if vector_dim <= 0:
        raise ValueError(f"Vector dimension must be > 0. Found: {vector_dim}")
        
    if len(metadata_df) == 0:
        raise ValueError("Metadata file is empty.")
        
    # Validation 3: Counts match
    if num_vectors != len(metadata_df):
        raise ValueError(f"Mismatch: {num_vectors} embeddings vs {len(metadata_df)} metadata rows.")
        
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Metadata shape: {metadata_df.shape}")
    logger.info(f"Vector dimension: {vector_dim}")
    
    # Initialize Qdrant Client
    Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Connecting to local Qdrant database at: {qdrant_path}")
    client = QdrantClient(path=qdrant_path)
    
    # Recreate Collection
    collection_creation_start = time.time()
    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' exists. Deleting...")
        client.delete_collection(collection_name)
        
    logger.info(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    collection_creation_time = time.time() - collection_creation_start
    
    # Prepare payload
    payloads = metadata_df.to_dict(orient="records")
    
    # Upload batches
    logger.info(f"Uploading {num_vectors} vectors in batches of {batch_size}...")
    points = []
    
    upload_start = time.time()
    for idx in tqdm(range(num_vectors), desc="Batch upload progress"):
        payload = {
            "product_id": payloads[idx].get("product_id"),
            "product_title": payloads[idx].get("product_title"),
            "product_brand": payloads[idx].get("product_brand"),
            "product_bullet_point": payloads[idx].get("product_bullet_point"),
            "product_description": payloads[idx].get("product_description"),
            "product_text": payloads[idx].get("product_text"),
            "split": payloads[idx].get("split"),
            "embedding_model": "BAAI/bge-small-en-v1.5"
        }
        
        point = PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload=payload
        )
        points.append(point)
        
        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []
            
    # Upload remainder
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    upload_time = time.time() - upload_start
        
    # Verify Upload
    verification_start = time.time()
    logger.info("Verifying upload...")
    if not client.collection_exists(collection_name):
        raise RuntimeError("Verification failed: Collection does not exist after upload.")
        
    collection_info = client.get_collection(collection_name)
    uploaded_count = collection_info.points_count
    
    if uploaded_count != num_vectors:
        raise RuntimeError(f"Verification failed: Expected {num_vectors} points, found {uploaded_count}.")
    verification_time = time.time() - verification_start
    
    total_runtime = time.time() - total_start_time
    
    # Improved Final Success Message
    logger.info("Verification successful.")
    logger.info("Qdrant collection verified successfully.")
    logger.info("All vectors indexed correctly.")
    logger.info("Semantic search database ready.")
    
    # Collection Statistics
    logger.info("----- Collection Statistics -----")
    logger.info(f"Collection Name : {collection_name}")
    logger.info(f"Points Count : {uploaded_count}")
    logger.info(f"Vector Dimension : {vector_dim}")
    logger.info("Distance Metric : COSINE")
    logger.info(f"Batch Size : {batch_size}")
    logger.info("Embedding Model : BAAI/bge-small-en-v1.5")
    
    # Runtime Tracking Display
    logger.info("----- Runtime Statistics -----")
    logger.info(f"Data Loading Time : {data_loading_time:.2f} sec")
    logger.info(f"Collection Creation Time : {collection_creation_time:.2f} sec")
    logger.info(f"Upload Time : {upload_time:.2f} sec")
    logger.info(f"Verification Time : {verification_time:.2f} sec")
    logger.info(f"Total Runtime : {total_runtime / 60:.2f} min")
    
    # Generate Report File
    report_path = Path("reports/qdrant_collection_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = f"""QDRANT COLLECTION REPORT
========================

Collection Name: {collection_name}
Points Count: {uploaded_count}
Vector Dimension: {vector_dim}
Distance Metric: COSINE
Embedding Model: BAAI/bge-small-en-v1.5
Batch Size: {batch_size}

Data Loading Time: {data_loading_time:.2f} sec
Collection Creation Time: {collection_creation_time:.2f} sec
Upload Time: {upload_time:.2f} sec
Verification Time: {verification_time:.2f} sec
Total Runtime: {total_runtime / 60:.2f} min
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

if __name__ == "__main__":
    load_embeddings_to_qdrant()
