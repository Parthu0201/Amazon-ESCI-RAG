import pandas as pd
import argparse
from pathlib import Path
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(filepath: Path) -> pd.DataFrame:
    """
    Load the dataset and validate it.
    
    Args:
        filepath (Path): Path to the parquet dataset.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the dataset is empty.
    """
    if not filepath.exists():
        logger.error(f"Dataset file not found: {filepath}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
    logger.info(f"Loading dataset from {filepath}")
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        logger.error(f"Failed to read parquet file: {e}")
        raise
        
    if df.empty:
        logger.error("Dataset is empty.")
        raise ValueError("Dataset is empty.")
        
    return df

def find_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Filter the dataset for the given query (case-insensitive).
    
    Args:
        df (pd.DataFrame): The dataset.
        query (str): The search query.
        
    Returns:
        pd.DataFrame: Filtered dataset containing only the matching query.
    """
    if 'query' not in df.columns:
        logger.error("'query' column not found in dataset.")
        return pd.DataFrame()
        
    # Case-insensitive exact match
    query_clean = query.strip().lower()
    return df[df['query'].astype(str).str.strip().str.lower() == query_clean].copy()

def group_by_esci(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group the filtered results by ESCI label.
    
    Args:
        df (pd.DataFrame): Filtered dataset.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with ESCI labels (E, S, C, I) as keys.
    """
    if 'esci_label' not in df.columns:
        logger.error("'esci_label' column not found.")
        return {}
        
    groups = {}
    for label in ['E', 'S', 'C', 'I']:
        groups[label] = df[df['esci_label'] == label].copy()
        
    return groups

def display_results(query: str, esci_groups: Dict[str, pd.DataFrame], output_str: List[str]) -> None:
    """
    Format and display the results to the console, and append to output string list.
    
    Args:
        query (str): The search query.
        esci_groups (Dict[str, pd.DataFrame]): Dictionary of dataframes grouped by ESCI label.
        output_str (List[str]): List to accumulate output lines for the report.
    """
    def add_line(line: str = ""):
        print(line)
        output_str.append(line)

    add_line("==================================================")
    add_line(f"QUERY: {query}")
    add_line("=====================")
    add_line()
    
    label_names = {
        'E': 'Exact Matches',
        'S': 'Substitutes',
        'C': 'Complements',
        'I': 'Irrelevant'
    }
    
    total_found = 0
    counts = {}
    
    for label in ['E', 'S', 'C', 'I']:
        group_df = esci_groups.get(label, pd.DataFrame())
        count = len(group_df)
        counts[label] = count
        total_found += count
        
        add_line(f"## {label} - {label_names[label]}")
        add_line()
        
        if count == 0:
            add_line("No products found.")
        else:
            for i, (_, row) in enumerate(group_df.iterrows(), 1):
                # Determine the best product text to display based on availability
                product_text = ""
                if 'product_text' in row and pd.notnull(row['product_text']) and str(row['product_text']).strip():
                    product_text = str(row['product_text'])
                elif 'product_title' in row and pd.notnull(row['product_title']) and str(row['product_title']).strip():
                    product_text = str(row['product_title'])
                elif 'product_name' in row and pd.notnull(row['product_name']) and str(row['product_name']).strip():
                    product_text = str(row['product_name'])
                else:
                    product_text = f"Product ID: {row.get('product_id', 'Unknown')}"
                
                # Truncate text if it's exceptionally long for console display
                display_text = (product_text[:120] + '...') if len(product_text) > 120 else product_text
                add_line(f"{i}. {display_text}")
                
        add_line()
        add_line(f"Total {label} Products: {count}")
        add_line()
        if label != 'I':
            add_line("---")
            add_line()

    add_line("==================================================")
    add_line()
    add_line(f"# Total Products Found: {total_found}")
    add_line()
    
    if total_found > 0:
        for label in ['E', 'S', 'C', 'I']:
            count = counts[label]
            add_line(f"{label} : {count}")
            
        add_line()
        
        for label in ['E', 'S', 'C', 'I']:
            count = counts[label]
            percentage = (count / total_found) * 100
            add_line(f"{label} : {count} ({percentage:.2f}%)")
    
    add_line()

def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the filtered dataset to CSV with specific columns.
    
    Args:
        df (pd.DataFrame): Filtered dataset.
        output_path (Path): Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Required columns
    target_cols = ['query', 'product_id', 'product_text', 'esci_label']
    
    # If product_text is missing, try to create it from fallback columns
    df_out = df.copy()
    if 'product_text' not in df_out.columns:
        if 'product_title' in df_out.columns:
            df_out['product_text'] = df_out['product_title']
        elif 'product_name' in df_out.columns:
            df_out['product_text'] = df_out['product_name']
        else:
            df_out['product_text'] = "N/A"
            
    # Add any missing required column with NA to avoid errors
    for col in target_cols:
        if col not in df_out.columns:
            df_out[col] = "N/A"

    try:
        df_out[target_cols].to_csv(output_path, index=False)
        logger.info(f"CSV report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")

def save_report(output_str: List[str], output_path: Path) -> None:
    """
    Save the generated report to a text file.
    
    Args:
        output_str (List[str]): List of output strings.
        output_path (Path): Destination text file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_str))
        logger.info(f"Text report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Amazon ESCI Query Lookup Tool.")
    parser.add_argument("--query", type=str, default= "iphone charger", help="Query to search for in the dataset.")
    args = parser.parse_args()
    
    query = args.query
    dataset_path = Path("data/processed/retrieval_dataset.parquet")
    reports_dir = Path("reports")
    
    csv_out = reports_dir / "query_esci_lookup.csv"
    txt_out = reports_dir / "query_esci_lookup.txt"
    
    try:
        df = load_dataset(dataset_path)
    except Exception:
        return
        
    query_df = find_query(df, query)
    
    if query_df.empty:
        print("No matching query found.")
        return
        
    esci_groups = group_by_esci(query_df)
    
    output_str = []
    display_results(query, esci_groups, output_str)
    
    save_csv(query_df, csv_out)
    save_report(output_str, txt_out)

if __name__ == "__main__":
    main()


# "iphone charger"
# "iphone 11 Charger"
# "power bank"
# "paper bags without handle"
# "#20 paper bags without handle"
# "revent 80 cfm"
