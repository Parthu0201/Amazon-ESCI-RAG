import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_product_embeddings(
    input_path: str = "data/samples/retrieval_sample_10000.parquet",
    embedding_output_path: str = "vectorstore/embeddings/product_embeddings.npy",
    metadata_output_path: str = "vectorstore/metadata/product_metadata.parquet",
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 128
) -> None:
    """
    Generates semantic product embeddings for FAISS retrieval using a pre-trained sentence transformer.
    
    Reads the retrieval dataset, removes duplicate products, generates embeddings for the
    product text, and saves both the embeddings as a NumPy array and the metadata as a Parquet file.
    
    Args:
        input_path (str): Path to the processed retrieval dataset.
        embedding_output_path (str): Destination path for the numpy embeddings file.
        metadata_output_path (str): Destination path for the parquet metadata file.
        model_name (str): The Hugging Face model identifier for SentenceTransformers.
        batch_size (int): Batch size for generating embeddings.
    """
    logger.info("Starting product embedding generation.")
    
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Missing retrieval dataset: {input_path}")
        
    try:
        # Load dataset
        logger.info(f"Loading retrieval dataset from {input_path}...")
        df = pd.read_parquet(input_path)
        logger.info(f"Initial dataset shape: {df.shape}")
        
        required_columns = [
            'product_id',
            'product_title',
            'product_text',
            'esci_label',
            'split'
        ]
        
        # Validate required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")
            
        # Keep only required columns to save memory
        df = df[required_columns]
        
        # Remove duplicate products
        logger.info("Removing duplicate products to keep one embedding per unique product...")
        unique_products_df = df.drop_duplicates(subset=['product_id']).reset_index(drop=True)
        unique_count = len(unique_products_df)
        logger.info(f"Unique product count: {unique_count}")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info(f"Using device: {model.device}")
        
        # Generate embeddings
        logger.info(f"Generating embeddings with batch size: {batch_size}")
        product_texts = unique_products_df['product_text'].tolist()
        
        embeddings = model.encode(
            product_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        embedding_dimension = embeddings.shape[1]
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        logger.info(f"Embedding dimension: {embedding_dimension}")
        
        # Save embeddings
        emb_out_file = Path(embedding_output_path)
        emb_out_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving embeddings to {embedding_output_path}...")
        np.save(emb_out_file, embeddings)
        
        # Save metadata
        meta_out_file = Path(metadata_output_path)
        meta_out_file.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_columns = ['product_id', 'product_title', 'esci_label', 'split']
        metadata_df = unique_products_df[metadata_columns]
        
        logger.info(f"Saving metadata to {metadata_output_path}...")
        metadata_df.to_parquet(meta_out_file, index=False)
        
        logger.info("Product embeddings and metadata generation completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred during embedding generation: {e}")
        raise

if __name__ == "__main__":
    generate_product_embeddings()
