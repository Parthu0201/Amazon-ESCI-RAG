import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Set, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.reranking.hybrid_rerank_search import hybrid_rerank_search
except ImportError as e:
    raise ImportError(f"Could not import hybrid_rerank_search from src.reranking.hybrid_rerank_search. Ensure you are running from the project root. Details: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ESCI_TO_SCORE = {
    "E": 3,
    "S": 2,
    "C": 1,
    "I": 0
}

def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Precision@K."""
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_ids]
    return len(relevant_retrieved) / k


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Recall@K."""
    if not relevant_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_ids]
    return len(relevant_retrieved) / len(relevant_ids)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Calculates Mean Reciprocal Rank (MRR)."""
    for i, doc in enumerate(retrieved_ids):
        if doc in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], gt_relevance: Dict[str, int], k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (nDCG@K).
    gt_relevance maps product_id to relevance score (0-3).
    """
    retrieved_k = retrieved_ids[:k]
    
    dcg = 0.0
    for i, doc in enumerate(retrieved_k):
        rel = gt_relevance.get(doc, 0)
        if rel > 0:
            dcg += ( (2 ** rel - 1) / np.log2(i + 2) )
            
    ideal_scores = sorted(list(gt_relevance.values()), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_scores):
        if rel > 0:
            idcg += ( (2 ** rel - 1) / np.log2(i + 2) )
            
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg


def evaluate_retrieval(
    ground_truth_path: str = "data/evaluation/ground_truth.parquet",
    sample_size: int = 100,
    top_k: int = 10
) -> None:
    """
    Evaluates the retrieval quality of the Qdrant vector database using the ground truth dataset.
    
    Args:
        ground_truth_path (str): Path to the ground truth parquet file.
        sample_size (int): Number of queries to randomly sample for evaluation.
        top_k (int): Number of documents to retrieve per query.
    """
    logger.info("Starting retrieval evaluation process...")
    logger.info("Retrieval Mode: Hybrid Search + Cross Encoder Re-ranking")
    
    gt_file = Path(ground_truth_path)
    if not gt_file.exists():
        logger.error(f"Ground truth file not found: {gt_file}")
        raise FileNotFoundError(f"Missing ground truth file at {gt_file}")
        
    logger.info(f"Loading ground truth from {gt_file}...")
    try:
        df = pd.read_parquet(gt_file)
    except Exception as e:
        logger.error(f"Failed to read parquet file: {e}")
        raise RuntimeError(f"Error loading ground truth dataset: {e}")
        
    required_columns = {"query", "product_id", "esci_label", "split"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logger.error(f"Missing required columns in dataset: {missing}")
        raise ValueError(f"Ground truth dataset is missing columns: {missing}")
        
    # IMPORTANT: Qdrant Product Filtering
    # We only have a subset of products in Qdrant (e.g. 10,000 samples). 
    # We must filter the ground truth so it only evaluates against products actually available in the database.
    logger.info("Loading Qdrant metadata to filter valid products...")
    try:
        metadata_df = pd.read_parquet("vectorstore/metadata/product_metadata.parquet")
        qdrant_product_ids = set(metadata_df["product_id"])
    except Exception as e:
        logger.error(f"Failed to read Qdrant metadata: {e}")
        raise RuntimeError(f"Error loading Qdrant metadata: {e}")
        
    original_gt_rows = len(df)
    df = df[df["product_id"].isin(qdrant_product_ids)].copy()
    filtered_gt_rows = len(df)
    removed_rows = original_gt_rows - filtered_gt_rows
    
    logger.info(f"Original ground truth rows: {original_gt_rows}")
    logger.info(f"Filtered ground truth rows: {filtered_gt_rows}")
    logger.info(f"Number of removed rows: {removed_rows}")
    
    if df.empty:
        logger.error("Ground truth dataset is empty after filtering for Qdrant product IDs.")
        raise ValueError("No valid products remaining in ground truth.")

    # Filter for test split
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        logger.error("No test data found in the ground truth dataset.")
        raise ValueError("The 'split' column has no 'test' values.")
        
    # Get unique queries
    unique_queries = test_df["query"].unique()
    if len(unique_queries) == 0:
        logger.error("No unique queries found in the test split.")
        raise ValueError("Query set is empty.")
        
    # Sample queries
    np.random.seed(42) # For reproducibility
    actual_sample_size = min(sample_size, len(unique_queries))
    logger.info(f"Randomly sampling {actual_sample_size} queries out of {len(unique_queries)} test queries...")
    sampled_queries = np.random.choice(unique_queries, size=actual_sample_size, replace=False)
    
    metrics = {
        "recall_at_k": [],
        "precision_at_k": [],
        "mrr": [],
        "ndcg_at_k": []
    }
    
    logger.info(f"Starting evaluation loop for Top-{top_k} retrieval...")
    
    for query in tqdm(sampled_queries, desc="Evaluating Queries"):
        # Build ground truth relevance for the query
        query_gt = test_df[test_df["query"] == query]
        
        if len(metrics["recall_at_k"]) < 10:
            print("\n" + "="*50)
            safe_query = query.encode("ascii", "replace").decode("ascii")
            print(f"Query: {safe_query}")
            print(f"Ground Truth Rows: {len(query_gt)}")
        
        # Relevant items are E, S, C
        relevant_ids = set(query_gt[query_gt["esci_label"].isin(["E", "S", "C"])]["product_id"])
        
        if len(metrics["recall_at_k"]) < 10:
            print(f"Relevant Products: {len(relevant_ids)}")
        
        # Relevance mapping for nDCG
        gt_relevance = {}
        for _, row in query_gt.iterrows():
            pid = row["product_id"]
            label = row["esci_label"]
            score = ESCI_TO_SCORE.get(label, 0)
            gt_relevance[pid] = score
            
        # Retrieve results
        try:
            results = hybrid_rerank_search(query=query, top_k=top_k, candidate_pool_size=20)
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            continue
            
        retrieved_ids = [res.get("product_id") for res in results if res.get("product_id")]
        
        if len(metrics["recall_at_k"]) < 10:
            print(f"Retrieved Products: {len(retrieved_ids)}")
            print(f"Top Retrieved IDs: {retrieved_ids[:5]}")
        
        # Calculate metrics
        recall = calculate_recall_at_k(retrieved_ids, relevant_ids, top_k)
        precision = calculate_precision_at_k(retrieved_ids, relevant_ids, top_k)
        mrr = calculate_mrr(retrieved_ids, relevant_ids)
        ndcg = calculate_ndcg_at_k(retrieved_ids, gt_relevance, top_k)
        
        metrics["recall_at_k"].append(recall)
        metrics["precision_at_k"].append(precision)
        metrics["mrr"].append(mrr)
        metrics["ndcg_at_k"].append(ndcg)
        
    if not metrics["recall_at_k"]:
        logger.error("Retrieval failed for all sampled queries. Result set is empty.")
        raise RuntimeError("No metrics could be calculated.")
        
    # Compute averages
    avg_recall = np.mean(metrics["recall_at_k"])
    avg_precision = np.mean(metrics["precision_at_k"])
    avg_mrr = np.mean(metrics["mrr"])
    avg_ndcg = np.mean(metrics["ndcg_at_k"])
    
    logger.info("Evaluation completed.")
    
    print("\nRetrieval Mode : Hybrid + Cross Encoder Re-ranking\n")
    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 28 + "\n")
    print(f"Queries Evaluated : {actual_sample_size}")
    print(f"Top-K             : {top_k}\n")
    print(f"Recall@{top_k:<10}: {avg_recall:.4f}")
    print(f"Precision@{top_k:<7}: {avg_precision:.4f}")
    print(f"MRR               : {avg_mrr:.4f}")
    print(f"nDCG@{top_k:<12}: {avg_ndcg:.4f}")
    print("=" * 50 + "\n")
    
    # Cleanly close Qdrant connection to avoid __del__ shutdown error
    try:
        from src.retrieval.search_qdrant import _client
        if _client is not None:
            _client.close()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        evaluate_retrieval()
    except Exception as err:
        logger.error(f"Retrieval evaluation failed: {err}")



