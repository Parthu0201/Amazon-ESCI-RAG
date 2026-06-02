import numpy as np
import pandas as pd
import logging
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
    
    # Validation 1: Files exist
    emb_file = Path(embeddings_path)
    meta_file = Path(metadata_path)
    
    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    # Load data
    logger.info("Loading embeddings and metadata...")
    embeddings = np.load(embeddings_path)
    metadata_df = pd.read_parquet(metadata_path)
    
    # Validation 2: Not empty
    if len(embeddings) == 0:
        raise ValueError("Embeddings file is empty.")
    if len(metadata_df) == 0:
        raise ValueError("Metadata file is empty.")
        
    # Validation 3: Counts match
    if len(embeddings) != len(metadata_df):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_df)} metadata rows.")
        
    num_vectors, vector_dim = embeddings.shape
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Metadata shape: {metadata_df.shape}")
    logger.info(f"Vector dimension: {vector_dim}")
    
    # Initialize Qdrant Client
    Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Connecting to local Qdrant database at: {qdrant_path}")
    client = QdrantClient(path=qdrant_path)
    
    # Recreate Collection
    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' exists. Deleting...")
        client.delete_collection(collection_name)
        
    logger.info(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    
    # Prepare payload
    payloads = metadata_df.to_dict(orient="records")
    
    # Upload batches
    logger.info(f"Uploading {num_vectors} vectors in batches of {batch_size}...")
    points = []
    
    for idx in tqdm(range(num_vectors), desc="Batch upload progress"):
        payload = {
            "product_id": payloads[idx].get("product_id"),
            "product_title": payloads[idx].get("product_title"),
            "esci_label": payloads[idx].get("esci_label"),
            "split": payloads[idx].get("split")
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
        
    # Verify Upload
    logger.info("Verifying upload...")
    if not client.collection_exists(collection_name):
        raise RuntimeError("Verification failed: Collection does not exist after upload.")
        
    collection_info = client.get_collection(collection_name)
    uploaded_count = collection_info.points_count
    
    if uploaded_count != num_vectors:
        raise RuntimeError(f"Verification failed: Expected {num_vectors} points, found {uploaded_count}.")
        
    logger.info(f"Verification successful: {uploaded_count} points uploaded correctly.")
    logger.info("Qdrant database is ready for semantic search!")

if __name__ == "__main__":
    load_embeddings_to_qdrant()
