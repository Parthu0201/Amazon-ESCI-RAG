import pandas as pd
import logging
from pathlib import Path
from load_data import load_esci_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_esci_datasets(examples_df: pd.DataFrame, products_df: pd.DataFrame, output_path: str = "data/processed/merged_dataset.csv") -> pd.DataFrame:
    """
    Merge examples and products datasets using product_id and product_locale.
    
    Args:
        examples_df (pd.DataFrame): Examples dataframe.
        products_df (pd.DataFrame): Products dataframe.
        output_path (str): Path to save the merged dataset.
        
    Returns:
        pd.DataFrame: Merged dataframe.
    """
    logger.info("Merging examples and products datasets...")
    
    # Check if necessary columns exist
    join_keys = ['product_id', 'product_locale']
    for key in join_keys:
        if key not in examples_df.columns or key not in products_df.columns:
            logger.error(f"Missing key '{key}' in datasets.")
            raise ValueError(f"Missing key '{key}' for merging.")
            
    try:
        merged_df = pd.merge(examples_df, products_df, on=join_keys, how='left')
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving merged dataset to {output_path}...")
        merged_df.to_csv(output_path, index=False)
        logger.info("Merge operation completed successfully.")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        raise

if __name__ == "__main__":
    examples, products, _ = load_esci_datasets()
    merge_esci_datasets(examples, products)
