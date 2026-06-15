import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """Configures logging for the script to output to stdout."""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

def _get_distribution_str(df: pd.DataFrame, col: str = "esci_label") -> str:
    """
    Calculates and formats the percentage distribution of a column.
    
    Args:
        df (pd.DataFrame): The dataframe containing the column.
        col (str): The column name to calculate the distribution for.
        
    Returns:
        str: A formatted string representing the distribution percentages.
    """
    counts = df[col].value_counts(normalize=True) * 100
    return "\n".join([f"{label} : {pct:.1f}%" for label, pct in counts.items()])

def create_training_sample(
    input_path: str = "data/processed/retrieval_dataset.parquet",
    train_size: int = 500000,
    test_size: int = 50000
) -> None:
    """
    Creates stratified training and test samples for the ESCI classification model.

    Args:
        input_path (str): Path to the processed retrieval dataset.
        train_size (int): Target number of rows for the training sample.
        test_size (int): Target number of rows for the test sample.
        
    Raises:
        FileNotFoundError: If the input parquet file does not exist.
        ValueError: If required columns are missing, dataset splits are insufficient,
                    or output datasets are empty.
        Exception: Reraises exceptions encountered during file reading/writing.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info("====================================")
    logger.info("ESCI TRAINING SAMPLE CREATION")
    logger.info("=============================\n")

    # Load dataset
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        logger.error(f"Failed to read parquet file {input_file}: {e}")
        raise

    # Validate columns
    required_columns = ["query", "product_text", "esci_label", "split"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log sizes
    original_size = len(df)
    logger.info(f"Original Dataset: {original_size} rows\n")

    logger.info("Split Distribution:")
    logger.info(df["split"].value_counts())
    logger.info("")

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_split_rows = len(train_df)
    test_split_rows = len(test_df)

    logger.info(f"Train Split Rows: {train_split_rows}")
    logger.info(f"Test Split Rows: {test_split_rows}\n")

    logger.info(f"Training Sample: {train_size} rows")
    logger.info(f"Test Sample: {test_size} rows\n")

    # Validate sizes against available rows
    if train_size > train_split_rows:
        error_msg = f"Requested train_size ({train_size}) exceeds available training rows ({train_split_rows})"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if test_size > test_split_rows:
        error_msg = f"Requested test_size ({test_size}) exceeds available test rows ({test_split_rows})"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log distribution before sampling
    logger.info("Label Distribution Before Sampling:")
    logger.info(_get_distribution_str(df, "esci_label") + "\n")

    # Create stratified samples
    train_sample, _ = train_test_split(
        train_df, 
        train_size=train_size, 
        stratify=train_df["esci_label"], 
        random_state=42
    )

    test_sample, _ = train_test_split(
        test_df,
        train_size=test_size,
        stratify=test_df["esci_label"],
        random_state=42
    )

    # Validate outputs are not empty
    if train_sample.empty or test_sample.empty:
        error_msg = "Output datasets are empty after sampling."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log distribution after sampling
    logger.info("Label Distribution After Sampling (Train):")
    logger.info(_get_distribution_str(train_sample, "esci_label") + "\n")

    logger.info("Label Distribution After Sampling (Test):")
    logger.info(_get_distribution_str(test_sample, "esci_label") + "\n")

    # Prepare save directories
    out_train_path = Path("data/classification/train_sample.parquet")
    out_test_path = Path("data/classification/test_sample.parquet")

    out_train_path.parent.mkdir(parents=True, exist_ok=True)
    out_test_path.parent.mkdir(parents=True, exist_ok=True)

    # Save outputs
    try:
        train_sample.to_parquet(out_train_path, index=False)
        test_sample.to_parquet(out_test_path, index=False)
    except Exception as e:
        logger.error(f"Failed to save output files: {e}")
        raise

    logger.info("Saved:")
    logger.info(out_train_path.as_posix())
    logger.info(out_test_path.as_posix())


if __name__ == "__main__":
    setup_logging()
    try:
        create_training_sample()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
