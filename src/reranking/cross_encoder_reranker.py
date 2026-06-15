import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Singleton cache for CrossEncoder model to avoid reloading."""
    _instance = None
    _model = None
    _metadata = None
    _load_time = 0.0
    _model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @classmethod
    def get_instance(cls, metadata_path: str = "vectorstore/metadata/product_metadata.parquet"):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_model(cls._model_name)
            cls._instance._load_metadata(metadata_path)
        return cls._instance

    def _load_model(self, model_name: str):
        logger.info(f"Loading Cross Encoder model: {model_name}...")
        start_time = time.time()
        try:
            self._model = CrossEncoder(model_name)
            self._load_time = time.time() - start_time
            logger.info("Cross Encoder Loaded Successfully")
            logger.info(f"Model Device: {self._model.model.device}")
        except Exception as e:
            logger.error(f"Failed to load Cross Encoder model: {e}")
            raise RuntimeError(f"Model loading error: {e}")

    def _load_metadata(self, metadata_path: str):
        try:
            path = Path(metadata_path)
            if path.exists():
                df = pd.read_parquet(metadata_path)
                df['product_id'] = df['product_id'].astype(str)
                self._metadata = dict(
                    zip(
                        df['product_id'],
                        df['product_text'].fillna("")
                    )
                )
            else:
                logger.warning(f"Metadata file not found at {metadata_path}. Will rely on provided product_text.")
        except Exception as e:
            logger.warning(f"Could not load metadata for product_text lookup: {e}")

    def rerank(self, query: str, retrieved_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not retrieved_results:
            return []
            
        if len(retrieved_results) < 10:
            logger.warning("Cross Encoder received fewer than 10 candidates.")
            
        start_time = time.time()
        
        # Prepare pairs
        pairs = []
        for res in retrieved_results:
            text = res.get("product_text")
            if not text and self._metadata:
                text = self._metadata.get(res["product_id"], "")
            if not text:
                text = res.get("product_title", "")
                
            pairs.append((query, text))
            
        logger.info(f"Pairs Generated: {len(pairs)}")
        
        # Generate Cross Encoder scores
        inf_start = time.time()
        try:
            cross_encoder_scores = self._model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Cross Encoder inference error: {e}")
            
        inf_time = time.time() - inf_start
        
        # Build output
        reranked_results = []
        for i, res in enumerate(retrieved_results):
            score = float(cross_encoder_scores[i])
            reranked_results.append({
                "product_id": res["product_id"],
                "product_title": res["product_title"],
                "vector_score": res.get("vector_score", 0.0),
                "bm25_score": res.get("bm25_score", 0.0),
                "hybrid_score": res.get("final_score", res.get("hybrid_score", 0.0)),
                "rerank_score": score
            })
            
        # Sort descending
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info(f"Products Re-ranked: {len(reranked_results)}")
        top_score = reranked_results[0]['rerank_score'] if reranked_results else 0.0
        
        if reranked_results:
            logger.info(f"Top Ranked Product: {reranked_results[0]['product_id']}")
            logger.info(f"Top Rerank Score: {top_score:.4f}")
        else:
            logger.info(f"Top Result Score: {top_score:.4f}")
        
        total_time = time.time() - start_time
        
        logger.info(f"Model Load Time : {self._load_time:.2f} sec")
        logger.info(f"Inference Time  : {inf_time:.2f} sec")
        logger.info(f"Total Runtime   : {total_time:.2f} sec")
        
        # Report generation
        report_path = Path("reports/cross_encoder_reranking_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_content = f"""CROSS ENCODER RERANKING REPORT
==============================

Model Name:
{self._model_name}

Query:
{query}

Products Reranked:
{len(reranked_results)}

Top Product ID:
{reranked_results[0]['product_id'] if reranked_results else 'N/A'}

Top Product Title:
{reranked_results[0]['product_title'] if reranked_results else 'N/A'}

Top Rerank Score:
{top_score:.2f}

Model Load Time:
{self._load_time:.2f} sec

Inference Time:
{inf_time:.2f} sec

Total Runtime:
{total_time:.2f} sec
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        return reranked_results

if __name__ == "__main__":
    from src.retrieval.search_hybrid import hybrid_search
    
    query = "wireless mouse"
    logger.info(f"Testing Cross Encoder Reranker with query: '{query}'")
    
    # Retrieve Hybrid Results
    top_k = 20
    results = hybrid_search(query, top_k=top_k)
    
    # Rerank
    reranker = CrossEncoderReranker.get_instance()
    reranked = reranker.rerank(query, results)
    
    print("\nTop 10 Re-ranked Results")
    print("-" * 50)
    for i, res in enumerate(reranked[:10]):
        title = res['product_title'][:50] + "..." if res['product_title'] and len(res['product_title']) > 50 else res['product_title']
        print(f"{i+1}. {title}")
        print(f"   ID: {res['product_id']}")
        print(f"   Scores -> Rerank: {res['rerank_score']:.4f} | Hybrid: {res['hybrid_score']:.4f}")
        print("-" * 50)
