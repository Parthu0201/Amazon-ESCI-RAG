import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_dataset(
    input_path: str = "data/processed/retrieval_dataset.parquet",
    output_path: str = "data/samples/retrieval_sample_50000.parquet",
    sample_size: int = 50000,
    random_seed: int = 42
) -> None:
    """
    Creates a sampled subset of the val dataset for fast testing and development.
    
    Reads the full retrieval dataset, removes duplicate products, and randomly
    samples the requested number of unique products. Saves the result as a new Parquet file.
    
    Args:
        input_path (str): The path to the full preprocessed retrieval dataset.
        output_path (str): The path where the sampled dataset will be saved.
        sample_size (int): The number of unique products to include in the sample.
        random_seed (int): The random seed for reproducibility.
    """
    logger.info("Starting sample dataset creation.")
    
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found at: {input_path}")
        raise FileNotFoundError(f"Missing input dataset: {input_path}")
        
    try:
        logger.info(f"Loading retrieval dataset from {input_path}...")
        df = pd.read_parquet(input_path)
        logger.info(f"Original dataset shape: {df.shape}")
        
        required_columns = ["product_id"]
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )
        
        # Remove duplicate products
        logger.info("Removing duplicate products by 'product_id'...")
        unique_products_df = df.drop_duplicates(subset=['product_id']).reset_index(drop=True)
        unique_count = len(unique_products_df)
        logger.info(f"Unique product count: {unique_count}")
        
        # Sample products
        if unique_count < sample_size:
            logger.warning(f"Requested sample size ({sample_size}) is larger than unique products ({unique_count}). Using all available unique products.")
            sampled_df = unique_products_df
        else:
            logger.info(f"Randomly sampling {sample_size} products with random_state={random_seed}...")
            sampled_df = unique_products_df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
            
        logger.info(f"Sample dataset shape: {sampled_df.shape}")
        
        # Save output
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving sample dataset to {output_path}...")
        sampled_df.to_parquet(output_path, index=False)
        logger.info("Sample dataset successfully created and saved.")
        
    except Exception as e:
        logger.error(f"An error occurred while creating the sample dataset: {e}")
        raise

if __name__ == "__main__":
    create_sample_dataset()
