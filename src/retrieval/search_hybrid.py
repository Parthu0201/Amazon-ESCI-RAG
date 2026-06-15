import pandas as pd
import logging
import time
import re
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

# Import existing dense retriever
from src.retrieval.search_qdrant import search_products

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

class BM25Retriever:
    """Singleton cache for BM25 index to avoid rebuilding on every query."""
    _instance = None
    _bm25 = None
    _metadata = None

    @classmethod
    def get_instance(cls, metadata_path: str = "vectorstore/metadata/product_metadata.parquet"):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_index(metadata_path)
        return cls._instance

    def _load_index(self, metadata_path: str):
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        logger.info(f"Loading metadata from {metadata_path} for BM25 index...")
        df = pd.read_parquet(metadata_path)
        
        # Ensure columns exist and fill NAs
        if 'product_text' not in df.columns:
            raise ValueError("product_text column missing in metadata.")
            
        df['product_text'] = df['product_text'].fillna("")
        df['product_id'] = df['product_id'].fillna("")
        df['product_title'] = df['product_title'].fillna("")
        
        self._metadata = df[['product_id', 'product_title', 'product_text']].to_dict('records')
        
        logger.info("Tokenizing documents for BM25...")
        tokenized_corpus = [re.findall(r"\w+", str(doc).lower()) for doc in df['product_text']]
        
        logger.info("Building BM25 index. This may take a moment...")
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self._bm25 or not self._metadata:
            raise RuntimeError("BM25 index is not initialized.")
            
        tokenized_query = re.findall(r"\w+", query.lower())
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top k indices efficiently using argsort
        top_n_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            results.append({
                "product_id": self._metadata[idx]["product_id"],
                "product_title": self._metadata[idx]["product_title"],
                "bm25_score": float(scores[idx])
            })
            
        return results

def normalize_scores(scores: List[float]) -> List[float]:
    """Min-Max normalization to safely map scores to 0.0 - 1.0 range."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.0 if max_score == 0 else 1.0 for _ in scores]
        
    return [(s - min_score) / (max_score - min_score) for s in scores]

def hybrid_search(query: str, top_k: int = 10, top_k_dense: int = 20, top_k_sparse: int = 20) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining Dense Qdrant search and Sparse BM25 search.
    """
    total_start_time = time.time()
    
    # 1. Dense Retrieval
    dense_start = time.time()
    try:
        dense_results_raw = search_products(query, top_k=top_k_dense)
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")
        dense_results_raw = []
        
    dense_results = []
    for res in dense_results_raw:
        dense_results.append({
            "product_id": res.get("product_id"),
            "product_title": res.get("product_title"),
            "vector_score": res.get("vector_score", 0.0)
        })
    dense_time = time.time() - dense_start
        
    # 2. Sparse Retrieval
    sparse_start = time.time()
    bm25_retriever = BM25Retriever.get_instance()
    sparse_results = bm25_retriever.search(query, top_k=top_k_sparse)
    sparse_time = time.time() - sparse_start
    
    # 3. Hybrid Score Fusion
    fusion_start = time.time()
    # Extract scores for normalization
    dense_scores = [r["vector_score"] for r in dense_results]
    sparse_scores = [r["bm25_score"] for r in sparse_results]
    
    norm_dense_scores = normalize_scores(dense_scores)
    norm_sparse_scores = normalize_scores(sparse_scores)
    
    # Apply normalized scores back
    for i, res in enumerate(dense_results):
        res["norm_vector_score"] = norm_dense_scores[i]
        
    for i, res in enumerate(sparse_results):
        res["norm_bm25_score"] = norm_sparse_scores[i]
        
    merged_dict = {}
    
    for res in dense_results:
        pid = res["product_id"]
        merged_dict[pid] = {
            "product_id": pid,
            "product_title": res["product_title"],
            "vector_score": res["vector_score"],
            "norm_vector_score": res["norm_vector_score"],
            "bm25_score": 0.0,
            "norm_bm25_score": 0.0
        }
        
    for res in sparse_results:
        pid = res["product_id"]
        if pid in merged_dict:
            merged_dict[pid]["bm25_score"] = res["bm25_score"]
            merged_dict[pid]["norm_bm25_score"] = res["norm_bm25_score"]
            # Fill title if it was missing in dense
            if not merged_dict[pid]["product_title"]:
                merged_dict[pid]["product_title"] = res["product_title"]
        else:
            merged_dict[pid] = {
                "product_id": pid,
                "product_title": res["product_title"],
                "vector_score": 0.0,
                "norm_vector_score": 0.0,
                "bm25_score": res["bm25_score"],
                "norm_bm25_score": res["norm_bm25_score"]
            }
            
    # Compute final score
    final_results = []
    for pid, res in merged_dict.items():
        final_score = (VECTOR_WEIGHT * res["norm_vector_score"]) + (BM25_WEIGHT * res["norm_bm25_score"])
        final_results.append({
            "product_id": res["product_id"],
            "product_title": res["product_title"],
            "vector_score": round(res["vector_score"], 4),
            "bm25_score": round(res["bm25_score"], 4),
            "final_score": round(final_score, 4)
        })
        
    # Final Ranking
    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = final_results[:top_k]
    fusion_time = time.time() - fusion_start
    total_time = time.time() - total_start_time
    
    # Logging requirements
    logger.info(f"Dense Results Retrieved: {len(dense_results)}")
    logger.info(f"Sparse Results Retrieved: {len(sparse_results)}")
    logger.info(f"Merged Results: {len(merged_dict)}")
    logger.info(f"Final Results Returned: {len(top_results)}")
    logger.info(f"Fusion Weights: Vector={VECTOR_WEIGHT} BM25={BM25_WEIGHT}")
    
    logger.info(f"Dense Retrieval Time : {dense_time:.2f} sec")
    logger.info(f"Sparse Retrieval Time: {sparse_time:.2f} sec")
    logger.info(f"Fusion Time          : {fusion_time:.2f} sec")
    logger.info(f"Total Search Time    : {total_time:.2f} sec")
    
    # Output to report
    report_path = Path("reports/hybrid_search_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_content = f"""HYBRID SEARCH REPORT
====================

Query: {query}
Dense Results Retrieved: {len(dense_results)}
Sparse Results Retrieved: {len(sparse_results)}
Merged Results: {len(merged_dict)}
Final Results Returned: {len(top_results)}

Vector Weight: {VECTOR_WEIGHT}
BM25 Weight: {BM25_WEIGHT}

Dense Retrieval Time : {dense_time:.2f} sec
Sparse Retrieval Time: {sparse_time:.2f} sec
Fusion Time          : {fusion_time:.2f} sec
Total Search Time    : {total_time:.2f} sec

"""
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(report_content)
    
    return top_results

if __name__ == "__main__":
    test_queries = [
        "wireless mouse",
        "iphone charger",
        "bluetooth speaker",
        "gaming keyboard"
    ]
    
    logger.info("Initializing Hybrid Search System...\n")
    
    report_file = Path("reports/hybrid_search_report.txt")
    if report_file.exists():
        report_file.unlink()
        
    for query in test_queries:
        logger.info(f"Query: '{query}'")
        logger.info("-" * 50)
        results = hybrid_search(query, top_k=5)
        logger.info("Results:")
        for i, res in enumerate(results):
            title = res['product_title'][:50] + "..." if res['product_title'] and len(res['product_title']) > 50 else res['product_title']
            logger.info(f"  {i+1}. {title}")
            logger.info(f"     ID: {res['product_id']}")
            logger.info(f"     Scores -> Vector: {res['vector_score']} | BM25: {res['bm25_score']} | Final: {res['final_score']}")
        logger.info("=" * 50)
