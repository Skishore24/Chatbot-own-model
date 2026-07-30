"""
backend/app/ai/rag/reranker.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Candidate Reranker Engine
Cross-Encoder style candidate scoring to re-order top-N retrieved passages.
"""

from typing import Dict, List, Tuple
from app.core.logger import logger


class CandidateReranker:
    """Enterprise Reranker Engine."""

    def rerank_passages(
        self, query: str, candidates: List[Dict[str, str]], top_n: int = 5
    ) -> List[Dict[str, str]]:
        """
        Reranks candidate document chunks based on n-gram token overlap and keyword density.
        """
        if not candidates:
            return []

        query_terms = set(query.lower().split())

        scored_candidates = []
        for cand in candidates:
            text = cand.get("text", "").lower()
            text_terms = text.split()

            if not text_terms:
                continue

            overlap = sum(1 for term in query_terms if term in text)
            exact_match_bonus = 2.0 if query.lower() in text else 0.0
            density_score = (overlap / max(len(text_terms), 1)) * 100.0

            final_score = cand.get("score", 0.0) + overlap + exact_match_bonus + density_score
            cand_copy = cand.copy()
            cand_copy["rerank_score"] = round(final_score, 4)
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_n]
