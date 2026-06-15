import pandas as pd
import numpy as np
import logging
import time
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
    input_path: str = "data/processed/retrieval_dataset.parquet",
    embedding_output_path: str = "vectorstore/embeddings/product_embeddings.npy",
    metadata_output_path: str = "vectorstore/metadata/product_metadata.parquet",
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256
) -> None:
    """
    Generates semantic product embeddings for Qdrant vector retrieval using a pre-trained sentence transformer.
    
    Reads the retrieval dataset, removes empty/duplicate products, generates normalized embeddings for the
    product text, and saves both the embeddings as a NumPy array and the metadata as a Parquet file.
    
    Args:
        input_path (str): Path to the processed retrieval dataset.
        embedding_output_path (str): Destination path for the numpy embeddings file.
        metadata_output_path (str): Destination path for the parquet metadata file.
        model_name (str): The Hugging Face model identifier for SentenceTransformers.
        batch_size (int): Batch size for generating embeddings.
    """
    logger.info("Starting product embedding generation.")
    total_start_time = time.time()
    
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Missing retrieval dataset: {input_path}")
        
    try:
        # Load dataset
        load_start_time = time.time()
        logger.info(f"Loading retrieval dataset from {input_path}...")
        df = pd.read_parquet(input_path)
        load_time = time.time() - load_start_time
        total_rows_loaded = len(df)
        logger.info(f"Initial dataset shape: {df.shape}")
        
        required_columns = [
            'product_id',
            'product_title',
            'product_brand',
            'product_bullet_point',
            'product_description',
            'product_text',
            'split'
        ]
        
        # Validate required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")
            
        # Keep only required columns to save memory
        df = df[required_columns]
        
        # Remove empty product text
        rows_before_filter = len(df)
        df = df[df["product_text"].fillna("").str.strip() != ""]
        rows_after_filter = len(df)
        rows_removed = rows_before_filter - rows_after_filter
        logger.info(f"Rows before filtering: {rows_before_filter}")
        logger.info(f"Rows after filtering: {rows_after_filter}")
        logger.info(f"Rows removed (empty product_text): {rows_removed}")
        
        # Remove duplicate products
        logger.info("Removing duplicate products to keep one embedding per unique product...")
        unique_products_df = df.drop_duplicates(subset=['product_id']).reset_index(drop=True)
        unique_count = len(unique_products_df)
        duplicates_removed = rows_after_filter - unique_count
        logger.info(f"Unique product count: {unique_count}")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Generate embeddings
        logger.info(f"Generating embeddings with batch size: {batch_size}")
        product_texts = unique_products_df['product_text'].tolist()
        
        embed_start_time = time.time()
        embeddings = model.encode(
            product_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embed_time = time.time() - embed_start_time
        
        embedding_dimension = embeddings.shape[1]
        
        logger.info("--- Embedding Statistics ---")
        logger.info(f"Model: {model_name}")
        logger.info(f"Dimension: {embedding_dimension}")
        logger.info(f"Products: {unique_count}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Device: {model.device}")
        
        # Metadata creation
        metadata_columns = ['product_id', 'product_title', 'product_brand', 'product_bullet_point', 'product_description', 'product_text', 'split']
        metadata_df = unique_products_df[metadata_columns]
        
        # Embedding Validation
        if embeddings.shape[0] != len(metadata_df):
            raise ValueError(f"Embedding count ({embeddings.shape[0]}) does not match metadata count ({len(metadata_df)})")
        
        save_start_time = time.time()
        
        # Save embeddings
        emb_out_file = Path(embedding_output_path)
        emb_out_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving embeddings to {embedding_output_path}...")
        np.save(emb_out_file, embeddings)
        
        # Save metadata
        meta_out_file = Path(metadata_output_path)
        meta_out_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving metadata to {metadata_output_path}...")
        metadata_df.to_parquet(meta_out_file, index=False)
        
        save_time = time.time() - save_start_time
        total_time = time.time() - total_start_time
        
        # Execution Timing Display
        logger.info("--- Execution Timing ---")
        logger.info(f"Dataset loading time: {load_time / 60:.2f} minutes")
        logger.info(f"Embedding Generation Time: {embed_time / 60:.2f} minutes")
        logger.info(f"Saving time: {save_time / 60:.2f} minutes")
        logger.info(f"Total Runtime: {total_time / 60:.2f} minutes")
        
        # Data Quality Report Display
        logger.info("--- Data Quality Report ---")
        logger.info(f"Total rows loaded: {total_rows_loaded}")
        logger.info(f"Unique products: {unique_count}")
        logger.info(f"Duplicate products removed: {duplicates_removed}")
        logger.info(f"Empty products removed: {rows_removed}")
        logger.info(f"Final products embedded: {unique_count}")
        
        # Save Summary Report
        report_path = Path("reports/embedding_generation_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_content = (
            f"Embedding Generation Report\n"
            f"===========================\n"
            f"Model Name: {model_name}\n"
            f"Embedding Dimension: {embedding_dimension}\n"
            f"Total Products: {total_rows_loaded}\n"
            f"Unique Products: {unique_count}\n"
            f"Empty Products Removed: {rows_removed}\n"
            f"Embedding Shape: {embeddings.shape}\n"
            f"Runtime: {total_time / 60:.2f} minutes\n"
        )
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        logger.info(f"Saved summary report to {report_path}")
        logger.info("Product embeddings and metadata generation completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred during embedding generation: {e}")
        raise

if __name__ == "__main__":
    generate_product_embeddings()

