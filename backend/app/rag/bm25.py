"""
backend/app/rag/bm25.py
----------------------------------------------------
Pure Python implementation of BM25 Okapi lexical scoring.
"""

import math
from typing import List, Tuple
from app.rag.index import InvertedIndex, normalize_tokens


class BM25Retriever:
    """Deterministic BM25 Okapi Ranking Engine."""

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self.idf_cache = self._precompute_idf()

    def _precompute_idf(self) -> dict:
        """Precomputes Inverse Document Frequency (IDF) for all terms."""
        idf = {}
        N = self.index.doc_count
        for term, df in self.index.df.items():
            # Standard Lucene/BM25 IDF formula with smoothing
            idf[term] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        return idf

    def score_query(self, query: str) -> List[float]:
        """Calculates BM25 scores for all documents given a query string."""
        scores = [0.0] * self.index.doc_count
        query_tokens = normalize_tokens(query)

        for token in query_tokens:
            if token not in self.index.postings:
                continue

            idf = self.idf_cache.get(token, 0.0)
            postings = self.index.postings[token]

            for doc_id, tf in postings.items():
                doc_len = self.index.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.index.avg_doc_len, 1e-6)))
                score_term = idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-6)
                scores[doc_id] += score_term

        return scores
