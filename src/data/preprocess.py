import pandas as pd
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(input_path: str = "data/processed/merged_dataset.csv", output_path: str = "data/processed/cleaned_dataset.csv") -> pd.DataFrame:
    """
    Preprocess the merged dataset.
    
    Args:
        input_path (str): Path to the merged dataset.
        output_path (str): Path to save the cleaned dataset.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    logger.info(f"Loading merged dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        
        initial_shape = df.shape
        logger.info(f"Initial shape: {initial_shape}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Shape after duplicate removal: {df.shape}")
        
        # Handle missing values
        # Drop rows where essential columns are missing
        essential_cols = ['query', 'product_title', 'esci_label']
        for col in essential_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        logger.info(f"Shape after handling missing essential values: {df.shape}")
        
        # Fill other missing values
        if 'product_description' in df.columns:
            df['product_description'] = df['product_description'].fillna('')
        if 'product_bullet_point' in df.columns:
            df['product_bullet_point'] = df['product_bullet_point'].fillna('')
            
        # Clean text columns
        text_cols = ['query', 'product_title', 'product_description', 'product_bullet_point']
        for col in text_cols:
            if col in df.columns:
                logger.info(f"Cleaning text in column: {col}...")
                df[col] = df[col].apply(clean_text)
                
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving cleaned dataset to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info("Preprocessing completed successfully.")
        
        return df
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    preprocess_dataset()
