import pandas as pd
import logging
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_esci_datasets(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all ESCI dataset files from the specified directory.
    
    Args:
        data_dir (str): Directory containing the raw dataset files.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Examples, Products, and Sources dataframes.
    """
    data_path = Path(data_dir)
    examples_path = data_path / "shopping_queries_dataset_examples.parquet"
    products_path = data_path / "shopping_queries_dataset_products.parquet"
    sources_path = data_path / "shopping_queries_dataset_sources.csv"
    
    for path in [examples_path, products_path, sources_path]:
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"Missing dataset file: {path}")
            
    try:
        logger.info(f"Loading examples from {examples_path}...")
        examples_df = pd.read_parquet(examples_path)
        logger.info(f"Examples shape: {examples_df.shape}")
        logger.info(f"Examples columns: {examples_df.columns.tolist()}")
        
        logger.info(f"Loading products from {products_path}...")
        products_df = pd.read_parquet(products_path)
        logger.info(f"Products shape: {products_df.shape}")
        logger.info(f"Products columns: {products_df.columns.tolist()}")
        
        logger.info(f"Loading sources from {sources_path}...")
        sources_df = pd.read_csv(sources_path)
        logger.info(f"Sources shape: {sources_df.shape}")
        logger.info(f"Sources columns: {sources_df.columns.tolist()}")
        
        return examples_df, products_df, sources_df
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

if __name__ == "__main__":
    load_esci_datasets()
