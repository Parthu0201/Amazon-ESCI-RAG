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

def build_qdrant_index(
    embeddings_path: str = "vectorstore/embeddings/product_embeddings.npy",
    metadata_path: str = "vectorstore/metadata/product_metadata.parquet",
    qdrant_path: str = "vectorstore/qdrant_db",
    collection_name: str = "esci_products",
    batch_size: int = 256
) -> None:
    """
    Builds a persistent local Qdrant vector database using pre-computed embeddings and metadata.
    
    Reads the NumPy embeddings and Parquet metadata, initializes a local Qdrant instance,
    creates the collection, and uploads the points with metadata payloads in batches.
    
    Args:
        embeddings_path (str): Path to the generated product embeddings numpy file.
        metadata_path (str): Path to the parquet metadata file.
        qdrant_path (str): Local path where Qdrant will store its persistence files.
        collection_name (str): The name of the Qdrant collection to create/overwrite.
        batch_size (int): The number of vectors to upload per API call.
    """
    logger.info("Starting Qdrant index build process.")
    
    # 1. Load data
    logger.info(f"Loading embeddings from {embeddings_path}...")
    if not Path(embeddings_path).exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    logger.info(f"Loading metadata from {metadata_path}...")
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata_df = pd.read_parquet(metadata_path)
    
    if len(embeddings) != len(metadata_df):
        raise ValueError(f"Mismatch between embeddings count ({len(embeddings)}) and metadata rows ({len(metadata_df)}).")
        
    num_vectors, vector_dim = embeddings.shape
    logger.info(f"Loaded {num_vectors} vectors of dimension {vector_dim}")
    
    # 2. Initialize Qdrant Client (Local File System)
    Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Initializing local Qdrant client at {qdrant_path}")
    client = QdrantClient(path=qdrant_path)
    
    # 3. Create Collection
    logger.info(f"Creating collection '{collection_name}' (recreating if it already exists)...")
    if client.collection_exists(collection_name):
        logger.warning(f"Collection '{collection_name}' already exists. Deleting it to start fresh.")
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    
    # 4. Upload Data in Batches
    logger.info("Uploading vectors and payloads to Qdrant...")
    points = []
    
    # Pre-convert metadata to a list of dicts for faster iteration
    payloads = metadata_df.to_dict(orient="records")
    
    for idx in tqdm(range(num_vectors), desc="Uploading points"):
        point = PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload=payloads[idx]
        )
        points.append(point)
        
        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []
            
    # Upload any remaining points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
    logger.info(f"Successfully uploaded {num_vectors} vectors to Qdrant collection '{collection_name}'.")

if __name__ == "__main__":
    build_qdrant_index()
