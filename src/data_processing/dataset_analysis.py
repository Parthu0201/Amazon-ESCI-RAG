import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(filepath: Path) -> pd.DataFrame:
    """
    Load the parquet dataset file and validate its existence.
    """
    if not filepath.exists():
        logger.error(f"Dataset file not found: {filepath}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    logger.info(f"Loading dataset from {filepath}")
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        logger.error(f"Error reading parquet file: {e}")
        raise
    
    if df.empty:
        logger.error("Dataset is empty.")
        raise ValueError("Dataset is empty.")
        
    return df

def analyze_dataset_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """Analyze the shape of the dataset."""
    rows, cols = df.shape
    return rows, cols

def analyze_unique_counts(df: pd.DataFrame) -> Tuple[int, int]:
    """Calculate unique counts for queries and product IDs."""
    unique_queries = df['query'].nunique() if 'query' in df.columns else 0
    unique_products = df['product_id'].nunique() if 'product_id' in df.columns else 0
    return unique_queries, unique_products

def analyze_split_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze the train/test split distribution."""
    if 'split' not in df.columns:
        return pd.DataFrame()
    
    counts = df['split'].value_counts()
    percentages = df['split'].value_counts(normalize=True) * 100
    
    dist_df = pd.DataFrame({
        'split': counts.index,
        'count': counts.values,
        'percentage': percentages.values
    })
    return dist_df

def analyze_esci_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze the ESCI label distribution."""
    if 'esci_label' not in df.columns:
        return pd.DataFrame()
    
    counts = df['esci_label'].value_counts()
    percentages = df['esci_label'].value_counts(normalize=True) * 100
    
    dist_df = pd.DataFrame({
        'esci_label': counts.index,
        'count': counts.values,
        'percentage': percentages.values
    })
    return dist_df

def analyze_missing_values(df: pd.DataFrame) -> pd.Series:
    """Count NULL values for all columns."""
    return df.isnull().sum()

def analyze_empty_values(df: pd.DataFrame) -> pd.Series:
    """Count empty strings or blank space strings for all object/string columns."""
    empties = {}
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            empties[col] = df[col].isin(["", " "]).sum()
    return pd.Series(empties)

def analyze_query_statistics(df: pd.DataFrame) -> Tuple[float, int, int]:
    """Calculate length statistics for the queries."""
    if 'query' not in df.columns:
        return 0.0, 0, 0
    
    lengths = df['query'].astype(str).apply(len)
    avg_len = lengths.mean()
    max_len = lengths.max()
    min_len = lengths.min()
    
    return avg_len, max_len, min_len

def analyze_product_statistics(df: pd.DataFrame) -> Tuple[float, int, int]:
    """Calculate length statistics for the product text."""
    if 'product_text' not in df.columns:
        return 0.0, 0, 0
    
    lengths = df['product_text'].astype(str).apply(len)
    avg_len = lengths.mean()
    max_len = lengths.max()
    min_len = lengths.min()
    
    return avg_len, max_len, min_len

def analyze_query_coverage(df: pd.DataFrame) -> Dict[str, int]:
    """Analyze query coverage of different ESCI label combinations."""
    if 'query' not in df.columns or 'esci_label' not in df.columns:
        return {}
        
    query_labels = df.groupby('query')['esci_label'].unique().apply(set)
    
    return {
        "Queries with only E": (query_labels == {'E'}).sum(),
        "Queries with E + S": (query_labels == {'E', 'S'}).sum(),
        "Queries with E + S + C": (query_labels == {'E', 'S', 'C'}).sum(),
        "Queries with E + S + C + I": (query_labels == {'E', 'S', 'C', 'I'}).sum()
    }

def analyze_products_per_query(df: pd.DataFrame) -> Tuple[float, int, int]:
    """Calculate statistics of products per query."""
    if 'query' not in df.columns:
        return 0.0, 0, 0
        
    sizes = df.groupby("query").size()
    return sizes.mean(), sizes.max(), sizes.min()

def analyze_esci_query_coverage(df: pd.DataFrame) -> Dict[str, int]:
    """Calculate unique queries containing each ESCI label."""
    if 'query' not in df.columns or 'esci_label' not in df.columns:
        return {}
        
    query_labels = df.groupby('query')['esci_label'].unique().apply(set)
    
    coverage = {}
    for label in ['E', 'S', 'C', 'I']:
        coverage[f"Queries containing {label}"] = query_labels.apply(lambda x: label in x).sum()
        
    return coverage

def generate_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save the comprehensive dataset report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / "dataset_report.txt"
    esci_file = output_dir / "esci_distribution.csv"
    split_file = output_dir / "split_distribution.csv"
    missing_file = output_dir / "missing_values.csv"
    empty_file = output_dir / "empty_values.csv"
    query_cov_file = output_dir / "query_coverage.csv"
    prod_per_query_file = output_dir / "products_per_query.csv"
    esci_cov_file = output_dir / "esci_query_coverage.csv"
    
    logger.info("Running analyses...")
    
    rows, cols = analyze_dataset_shape(df)
    unique_queries, unique_products = analyze_unique_counts(df)
    split_dist = analyze_split_distribution(df)
    esci_dist = analyze_esci_distribution(df)
    top_queries = df['query'].value_counts().head(20) if 'query' in df.columns else pd.Series(dtype=int)
    
    missing_vals = analyze_missing_values(df)
    empty_vals = analyze_empty_values(df)
    
    avg_q, max_q, min_q = analyze_query_statistics(df)
    avg_p, max_p, min_p = analyze_product_statistics(df)
    
    query_cov = analyze_query_coverage(df)
    avg_ppq, max_ppq, min_ppq = analyze_products_per_query(df)
    esci_cov = analyze_esci_query_coverage(df)
    
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    logger.info("Saving CSV artifacts...")
    if not split_dist.empty:
        split_dist.to_csv(split_file, index=False)
    if not esci_dist.empty:
        esci_dist.to_csv(esci_file, index=False)
    missing_vals.to_csv(missing_file, header=["count"], index_label="column")
    empty_vals.to_csv(empty_file, header=["count"], index_label="column")
    
    pd.DataFrame(list(query_cov.items()), columns=["coverage_type", "count"]).to_csv(query_cov_file, index=False)
    
    pd.DataFrame([{
        "average_products_per_query": avg_ppq,
        "maximum_products_per_query": max_ppq,
        "minimum_products_per_query": min_ppq
    }]).to_csv(prod_per_query_file, index=False)
    
    pd.DataFrame(list(esci_cov.items()), columns=["coverage_type", "count"]).to_csv(esci_cov_file, index=False)

    logger.info("Writing dataset report to text file...")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("AMAZON ESCI DATASET ANALYSIS\n")
        f.write("============================\n\n")
        
        # 1. Dataset Shape
        f.write("## Dataset Shape\n\n")
        f.write(f"Rows: {rows:,}\n")
        f.write(f"Columns: {cols:,}\n\n")
        
        # 2. Unique Counts
        f.write("## Unique Counts\n\n")
        f.write(f"Unique Queries : {unique_queries:,}\n")
        f.write(f"Unique Products: {unique_products:,}\n\n")
        
        # 3. Train/Test Split
        f.write("## Train/Test Split\n\n")
        if not split_dist.empty:
            for _, row in split_dist.iterrows():
                split_name = str(row['split']).capitalize()
                f.write(f"{split_name:<5} : {row['count']:>9,} ({row['percentage']:.2f}%)\n")
        else:
            f.write("N/A\n")
        f.write("\n")
        
        # 4. ESCI Distribution
        f.write("## ESCI Distribution\n\n")
        if not esci_dist.empty:
            for _, row in esci_dist.iterrows():
                f.write(f"{str(row['esci_label']):<1} : {row['count']:>9,} ({row['percentage']:.2f}%)\n")
        else:
            f.write("N/A\n")
        f.write("\n")
        
        # 5. Top Queries
        f.write("## Top Queries\n\n")
        for query, count in top_queries.items():
            f.write(f"{query}: {count:,}\n")
        f.write("\n")
        
        # 6. Missing Values
        f.write("## Missing Values (NULL)\n\n")
        max_col_len = max([len(str(c)) for c in missing_vals.index]) if not missing_vals.empty else 20
        for col_name, count_val in missing_vals.items():
            f.write(f"{str(col_name):<{max_col_len}} : {count_val:,}\n")
        f.write("\n")
        
        # 7. Empty String Values
        f.write("## Empty String Analysis\n\n")
        max_col_len_emp = max([len(str(c)) for c in empty_vals.index]) if not empty_vals.empty else 20
        for col_name, count_val in empty_vals.items():
            f.write(f"{str(col_name):<{max_col_len_emp}} : {count_val:,}\n")
        f.write("\n")
        
        # 8. Query Statistics
        f.write("## Query Statistics\n\n")
        f.write(f"Average Query Length : {avg_q:,.2f}\n")
        f.write(f"Maximum Query Length : {max_q:,}\n")
        f.write(f"Minimum Query Length : {min_q:,}\n\n")
        
        # 9. Product Statistics
        f.write("## Product Statistics\n\n")
        f.write(f"Average Product Length : {avg_p:,.2f}\n")
        f.write(f"Maximum Product Length : {max_p:,}\n")
        f.write(f"Minimum Product Length : {min_p:,}\n\n")
        
        # 10. Query Coverage Analysis
        f.write("## Query Coverage Analysis\n\n")
        for cov_type, cnt in query_cov.items():
            f.write(f"{cov_type:<27} : {cnt:,}\n\n")
            
        # 11. Products Per Query
        f.write("## Products Per Query\n\n")
        f.write(f"Average Products Per Query : {avg_ppq:,.2f}\n")
        f.write(f"Maximum Products Per Query : {max_ppq:,}\n")
        f.write(f"Minimum Products Per Query : {min_ppq:,}\n\n")
        
        # 12. ESCI Query Coverage
        f.write("## ESCI Query Coverage\n\n")
        for cov_type, cnt in esci_cov.items():
            f.write(f"{cov_type:<20} : {cnt:,}\n\n")
            
        # 13. Dataset Memory Usage
        f.write("## Dataset Memory Usage\n\n")
        f.write(f"{memory_usage_mb:,.2f} MB\n\n")
        
        # 14. Data Quality Notes
        f.write("## Data Quality Notes\n\n")
        f.write("The retrieval_dataset.parquet file is a processed dataset.\n")
        f.write("Missing values were replaced during preprocessing using fillna(\"\").\n\n")
        f.write("Therefore:\n\n")
        f.write("# Missing Values (NULL)\n")
        f.write("Current NULL values in the processed dataset.\n\n")
        f.write("# Empty String Values\n")
        f.write("Fields where missing information was converted into empty strings.\n\n")
        f.write("A value of 0 in the Missing Values section does not imply that the original source data contained no missing information.\n\n")
        
        f.write("==================================================\n")
        
    logger.info("Report generation complete.")

def main():
    """Main execution block."""
    processed_filepath = Path("data/processed/retrieval_dataset.parquet")
    output_dir = Path("reports")
    
    try:
        df = load_dataset(processed_filepath)
        generate_report(df, output_dir)
        
        # Log success and verify report exists
        report_file = output_dir / "dataset_report.txt"
        if report_file.exists():
            logger.info(f"Successfully generated report at {report_file}")
                
    except Exception as e:
        logger.error(f"Failed to analyze dataset: {e}")
        raise

if __name__ == "__main__":
    main()
