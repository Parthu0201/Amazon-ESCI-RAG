import pandas as pd
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_retrieval_dataset(
    input_path: str = "data/processed/cleaned_dataset.csv",
    output_path: str = "data/processed/retrieval_dataset.parquet"
) -> Optional[pd.DataFrame]:
    """
    Creates a retrieval dataset specifically for FAISS semantic search and embedding generation.
    
    Loads the cleaned dataset, filters the required columns, handles missing values,
    creates a consolidated product_text column, logs various statistics,
    and saves the output to a new Parquet file.
    
    Args:
        input_path (str): The file path to the input cleaned dataset CSV.
        output_path (str): The file path to save the output retrieval dataset Parquet.
        
    Returns:
        Optional[pd.DataFrame]: The resulting dataframe if successful, None otherwise.
        
    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If any of the required columns are missing in the dataset.
    """
    logger.info("Starting creation of retrieval dataset.")
    
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found at: {input_path}")
        raise FileNotFoundError(f"Missing input dataset: {input_path}")
        
    try:
        logger.info(f"Loading dataset from {input_path}...")
        # Read low_memory=False to handle mixed types gracefully if any exist
        df = pd.read_csv(input_path, low_memory=False)
        
        required_columns = [
            'query',
            'product_id',
            'product_title',
            'product_brand',
            'product_bullet_point',
            'product_description',
            'esci_label',
            'split'
        ]
        
        # Validate that all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")
            
        # Keep only the requested columns
        retrieval_df = df[required_columns].copy()
        
        # Fill missing text columns
        text_cols = ['product_title', 'product_brand', 'product_bullet_point', 'product_description', 'query']
        for col in text_cols:
            retrieval_df[col] = retrieval_df[col].fillna('').astype(str)
            
        # Create product_text column
        logger.info("Creating consolidated product_text column...")
        retrieval_df['product_text'] = (
            retrieval_df['product_title'].str.strip() + " " +
            retrieval_df['product_brand'].str.strip() + " " +
            retrieval_df['product_bullet_point'].str.strip() + " " +
            retrieval_df['product_description'].str.strip()
        )
        
        # Normalize whitespace in product_text
        retrieval_df['product_text'] = retrieval_df['product_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Remove duplicates
        retrieval_df = retrieval_df.drop_duplicates()
        
        # Reorder columns
        final_columns = [
            'query',
            'product_id',
            'product_title',
            'product_brand',
            'product_bullet_point',
            'product_description',
            'product_text',
            'esci_label',
            'split'
        ]
        retrieval_df = retrieval_df[final_columns]
        
        # Log Dataset Statistics
        logger.info(f"Retrieval dataset shape: {retrieval_df.shape}")
        logger.info(f"Retrieval dataset columns: {retrieval_df.columns.tolist()}")
        
        missing_values = retrieval_df.isnull().sum()
        logger.info(f"Missing values:\n{missing_values.to_string()}")
        
        label_distribution = retrieval_df['esci_label'].value_counts(normalize=True) * 100
        logger.info(f"ESCI label distribution (%):\n{label_distribution.to_string()}")
        
        split_distribution = retrieval_df['split'].value_counts(normalize=True) * 100
        logger.info(f"Split distribution (%):\n{split_distribution.to_string()}")
        
        unique_products = retrieval_df['product_id'].nunique()
        logger.info(f"Number of unique products: {unique_products}")
        
        unique_queries = retrieval_df['query'].nunique()
        logger.info(f"Number of unique queries: {unique_queries}")
        
        # Save output
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving retrieval dataset to {output_path}...")
        retrieval_df.to_parquet(output_path, index=False)
        logger.info("Retrieval dataset successfully created and saved.")
        
        return retrieval_df

    except Exception as e:
        logger.error(f"An error occurred while creating the retrieval dataset: {e}")
        raise

if __name__ == "__main__":
    create_retrieval_dataset()
