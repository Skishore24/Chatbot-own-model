"""
backend/app/rag/tfidf.py
----------------------------------------------------
Pure Python TF-IDF Vectorizer & Cosine Similarity search.
"""

import math
from typing import Dict, List
from app.rag.index import InvertedIndex, normalize_tokens


class TFIDFRetriever:
    """TF-IDF Vector Space Model with Cosine Similarity."""

    def __init__(self, index: InvertedIndex):
        self.index = index
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self.doc_norms: List[float] = []

        self._build_vectors()

    def _build_vectors(self) -> None:
        N = self.index.doc_count
        # Compute smoothed IDF
        for term, df in self.index.df.items():
            self.idf[term] = math.log((1.0 + N) / (1.0 + df)) + 1.0

        # Build document sparse vectors
        self.doc_vectors = [{} for _ in range(N)]
        for term, postings in self.index.postings.items():
            term_idf = self.idf.get(term, 1.0)
            for doc_id, tf in postings.items():
                # Sublinear term frequency scaling
                w = (1.0 + math.log(tf)) * term_idf
                self.doc_vectors[doc_id][term] = w

        # Precompute L2 norms for cosine normalization
        self.doc_norms = [0.0] * N
        for doc_id in range(N):
            norm_sq = sum(v * v for v in self.doc_vectors[doc_id].values())
            self.doc_norms[doc_id] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    def score_query(self, query: str) -> List[float]:
        """Computes Cosine Similarity between query vector and all documents."""
        scores = [0.0] * self.index.doc_count
        query_tokens = normalize_tokens(query)
        if not query_tokens:
            return scores

        # Build query vector
        query_tf: Dict[str, int] = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1

        query_vec: Dict[str, float] = {}
        query_norm_sq = 0.0
        for token, count in query_tf.items():
            if token in self.idf:
                w = (1.0 + math.log(count)) * self.idf[token]
                query_vec[token] = w
                query_norm_sq += w * w

        query_norm = math.sqrt(query_norm_sq) if query_norm_sq > 0 else 1.0

        # Sparse dot product
        for token, q_weight in query_vec.items():
            if token not in self.index.postings:
                continue
            for doc_id, _ in self.index.postings[token].items():
                d_weight = self.doc_vectors[doc_id].get(token, 0.0)
                scores[doc_id] += q_weight * d_weight

        # Normalize by product of norms
        for doc_id in range(self.index.doc_count):
            scores[doc_id] = scores[doc_id] / max(query_norm * self.doc_norms[doc_id], 1e-6)

        return scores
