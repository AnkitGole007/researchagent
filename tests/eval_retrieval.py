import os
import sys
import unittest
import numpy as np

# Adjust imports safely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import (
    load_bm25_index, 
    load_precomputed_embeddings, 
    select_embedding_candidates, 
    scibert_classify_papers, 
    fetch_from_sqlite,
    get_corpus_dir
)

class TestRetrievalPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Load full input pool
        cls.all_papers = fetch_from_sqlite("")
        print(f"Loaded {len(cls.all_papers)} papers for eval suite.")
        
        # 2. Build or load realistic test queries 
        # Using 25 realistic proxies if no static dataset exists. We assume the paper itself MUST be retrieved.
        # Format: (Query text, Expected arxiv_id)
        cls.test_cases = []
        if cls.all_papers:
            np.random.seed(42)
            sample_papers = np.random.choice(cls.all_papers, min(25, len(cls.all_papers)), replace=False)
            for p in sample_papers:
                # Build a realistic simulated research brief based on the title + abstract
                brief = f"I am looking for papers about {p.title}. {p.abstract[:200]}"
                cls.test_cases.append((brief, p.arxiv_id))
    
    def test_pipeline_metrics(self):
        if not self.all_papers:
            self.skipTest("No corpus available to test against.")
            return

        total_queries = len(self.test_cases)
        recalls_150 = []
        recalls_primary = []
        rr_10 = []
        ndcg_10 = []
        primary_rates = []

        print(f"\n--- Running Evaluation Harness on {total_queries} Queries ---")

        for query_brief, target_id in self.test_cases:
            # Stage 1-3: retrieve top 150 candidates
            candidates = select_embedding_candidates(
                self.all_papers, 
                query_brief=query_brief, 
                provider="groq", # Simulate groq/free path
                max_candidates=150,
                use_hyde=False
            )
            
            # Stage 4: simulate scibert threshold classification (Task 41)
            classified = scibert_classify_papers(candidates)
            primary_papers = [p for p in classified if getattr(p, "focus_label", "off-topic") == "primary"]
            
            primary_rates.append(len(primary_papers) / 150.0)

            # --- Check Retrieval Status ---
            retrieved_ranks = [i+1 for i, p in enumerate(candidates) if p.arxiv_id == target_id]
            rank = retrieved_ranks[0] if retrieved_ranks else -1
            
            # Recall@150
            is_recalled = 1 if rank > 0 else 0
            recalls_150.append(is_recalled)
            
            # MRR@10
            rr = 1.0 / rank if (0 < rank <= 10) else 0.0
            rr_10.append(rr)
            
            # NDCG@10 (binary relevance where ideal DCG = 1.0)
            ndcg = (1.0 / np.log2(rank + 1)) if (0 < rank <= 10) else 0.0
            ndcg_10.append(ndcg)
            
            # Recall@primary (Task 41 calibration metric)
            is_primary = any(p.arxiv_id == target_id for p in primary_papers)
            recalls_primary.append(1 if is_primary else 0)

        # Compute averages
        mean_recall_150 = np.mean(recalls_150)
        mean_recall_primary = np.mean(recalls_primary)
        mean_rr_10 = np.mean(rr_10)
        mean_ndcg_10 = np.mean(ndcg_10)
        mean_primary_rate = np.mean(primary_rates)
        
        print(f"Recall@150:    {mean_recall_150:.4f}")
        print(f"Recall@primary:{mean_recall_primary:.4f}")
        print(f"MRR@10:        {mean_rr_10:.4f}")
        print(f"NDCG@10:       {mean_ndcg_10:.4f}")
        print(f"Primary Rate:  {mean_primary_rate:.4f} (target: 0.10 - 0.35)")

        # Assertions mapping to implementation_order.md Gate
        self.assertGreaterEqual(
            mean_recall_150, 0.75, 
            "Recall@150 below 0.75! Stage 1 BM25/FAISS quality may be degrading."
        )
        self.assertGreaterEqual(
            mean_recall_primary, 0.85, 
            "Recall@primary strictly below 0.85! T-41 classification thresholds need calibrating."
        )
        self.assertTrue(
            0.10 <= mean_primary_rate <= 0.35, 
            f"Primary Rate {mean_primary_rate} is outside the [0.10, 0.35] target envelope. T-41 needs calibration."
        )

if __name__ == "__main__":
    unittest.main()
