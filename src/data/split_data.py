import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_dataset(input_path: str = "data/processed/cleaned_dataset.csv", output_dir: str = "data/processed", test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        input_path (str): Path to the cleaned dataset.
        output_dir (str): Directory to save the split datasets.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
    """
    logger.info(f"Loading cleaned dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        
        logger.info("Splitting dataset...")
        # First split into train_val and test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["esci_label"]
        )
        
        # Then split train_val into train and validation
        # Adjust val_size relative to the train_val portion
        relative_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=train_val_df["esci_label"]
        )
        
        logger.info(f"Train set shape: {train_df.shape}")
        logger.info(f"Validation set shape: {val_df.shape}")
        logger.info(f"Test set shape: {test_df.shape}")
        
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = out_dir / "train_dataset.csv"
        val_path = out_dir / "val_dataset.csv"
        test_path = out_dir / "test_dataset.csv"
        
        logger.info(f"Saving splits to {output_dir}...")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info("Data splitting completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise

if __name__ == "__main__":
    split_dataset()
