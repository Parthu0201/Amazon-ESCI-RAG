import time
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.retrieval.search_hybrid import hybrid_search
from src.reranking.cross_encoder_reranker import CrossEncoderReranker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def hybrid_rerank_search(
    query: str,
    top_k: int = 10,
    candidate_pool_size: int = 20
) -> List[Dict[str, Any]]:
    """
    Orchestrator for Hybrid Search followed by Cross Encoder Re-ranking.
    """
    total_start_time = time.time()
    
    # --- Step 1: Hybrid Search ---
    logger.info("Starting Hybrid Retrieval...")
    hybrid_start = time.time()
    try:
        hybrid_results = hybrid_search(
            query=query,
            top_k=candidate_pool_size
        )
    except Exception as e:
        logger.error(f"Hybrid Search Failure: {e}")
        return []
    hybrid_time = time.time() - hybrid_start
    
    logger.info(f"Hybrid Results Retrieved: {len(hybrid_results)}")
    
    if not hybrid_results:
        logger.warning("Empty Retrieval Results from Hybrid Search.")
        return []

    # --- Step 2 & 3: Cross Encoder Re-ranking ---
    logger.info("Starting Cross Encoder Re-ranking...")
    rerank_start = time.time()
    try:
        reranker = CrossEncoderReranker.get_instance()
        reranked_results = reranker.rerank(
            query=query,
            retrieved_results=hybrid_results
        )
    except Exception as e:
        logger.error(f"Cross Encoder Failure: {e}")
        return []
    rerank_time = time.time() - rerank_start
    
    logger.info(f"Candidates Re-ranked: {len(reranked_results)}")
    
    if not reranked_results:
        logger.warning("Empty Retrieval Results after Cross Encoder Re-ranking.")
        return []

    # --- Step 4: Final Results ---
    final_results = reranked_results[:top_k]
    logger.info(f"Final Results Returned: {len(final_results)}")
    
    total_time = time.time() - total_start_time
    
    # Log Runtimes
    logger.info(f"Hybrid Search Time : {hybrid_time:.2f} sec")
    logger.info(f"Re-ranking Time    : {rerank_time:.2f} sec")
    logger.info(f"Total Runtime      : {total_time:.2f} sec")
    
    # Report Generation
    report_path = Path("reports/hybrid_rerank_search_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    top_product_id = final_results[0]['product_id'] if final_results else "N/A"
    top_product_title = final_results[0]['product_title'] if final_results else "N/A"
    top_rerank_score = final_results[0].get('rerank_score', 0.0) if final_results else 0.0
    
    report_content = f"""HYBRID RERANK SEARCH REPORT
===========================

Query: {query}
Candidate Pool Size: {candidate_pool_size}
Final Top K: {top_k}

Hybrid Search Time: {hybrid_time:.2f} sec
Reranking Time: {rerank_time:.2f} sec
Total Runtime: {total_time:.2f} sec

Top Product ID: {top_product_id}
Top Product Title: {top_product_title}
Top Rerank Score: {top_rerank_score:.2f}

"""
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(report_content)
        
    return final_results

if __name__ == "__main__":
    query = "wireless mouse"
    
    # Clear the report file if it exists at the start of a test run
    report_file = Path("reports/hybrid_rerank_search_report.txt")
    if report_file.exists():
        report_file.unlink()
        
    print(f"Testing Hybrid Rerank Search for query: '{query}'\n")
    
    results = hybrid_rerank_search(
        query=query,
        top_k=10,
        candidate_pool_size=20
    )
    
    print("\nTop 10 Re-ranked Results")
    print("-" * 50)
    for i, res in enumerate(results):
        title = res['product_title'][:50] + "..." if res['product_title'] and len(res['product_title']) > 50 else res['product_title']
        print(f"Rank: {i+1}")
        print(f"Product ID: {res['product_id']}")
        print(f"Product Title: {title}")
        print(f"Rerank Score: {res['rerank_score']:.4f}")
        print("-" * 50)
