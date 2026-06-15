import logging
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # To match the exact example output logs
)
logger = logging.getLogger(__name__)

def create_ground_truth(
    input_path: str = "data/processed/retrieval_dataset.parquet",
    output_path: str = "data/evaluation/ground_truth.parquet"
) -> None:
    """
    Creates a ground truth dataset for retrieval evaluation from the full dataset.
    
    This function loads the retrieval dataset, validates that all necessary columns
    for evaluation exist, extracts only the required columns, removes duplicate rows,
    and saves the resulting lightweight dataframe to a new parquet file.
    
    Args:
        input_path (str): Path to the processed retrieval dataset parquet file.
        output_path (str): Path where the ground truth parquet file will be saved.
        
    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If required columns are missing or if the output dataframe is empty.
    """
    logger.info("Starting ground truth creation...")
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # 1. Input file validation
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file does not exist at {input_file}")
        
    # 2. Load dataset
    logger.info("Loading retrieval dataset...")
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        logger.error(f"Failed to load parquet file: {e}")
        raise RuntimeError(f"Error reading {input_file}: {e}")
        
    original_shape = df.shape
    logger.info(f"Original dataset shape: {original_shape}")
    
    # 3. Validate required columns
    required_columns = ["query", "product_id", "esci_label", "split"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Input dataset is missing required columns: {missing_columns}")
        
    # 4. Create new dataframe with ONLY the required columns
    gt_df = df[required_columns].copy()
    
    shape_before_dedup = gt_df.shape
    logger.info(f"Ground truth shape before deduplication: {shape_before_dedup}")
    
    # 5. Remove duplicate rows
    gt_df.drop_duplicates(inplace=True)
    
    shape_after_dedup = gt_df.shape
    logger.info(f"Ground truth shape after deduplication: {shape_after_dedup}")
    
    # Validation: Output dataframe is not empty
    if gt_df.empty:
        logger.error("Output ground truth dataframe is completely empty.")
        raise ValueError("Ground truth dataframe became empty after dropping duplicates.")
        
    # 6. Create directories automatically if they do not exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 7. Save output
    logger.info(f"Saving ground truth to {output_path}")
    try:
        gt_df.to_parquet(output_file, index=False)
    except Exception as e:
        logger.error(f"Failed to save ground truth dataset: {e}")
        raise RuntimeError(f"Error saving to {output_file}: {e}")
        
    logger.info("Ground truth creation completed successfully.")


if __name__ == "__main__":
    try:
        create_ground_truth()
    except Exception as err:
        logger.error(f"Ground truth creation failed: {err}")
