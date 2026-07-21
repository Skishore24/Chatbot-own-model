"""
ai/rag/retriever.py
----------------------------------------------------
Genkit AI - Advanced Hybrid Retriever (v4.0)

Features
--------
• Custom in-memory BM25 ranker
• Custom vector cosine similarity
• Intent-based relevance filtering & boosts
• Structured document chunk parsing

Author : Genkit AI
"""

import os
import re
import sys
import math
from collections import Counter
from typing import List, Dict, Optional

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.embeddings.embedding import _store
from config import logger

# ==========================================================
# IN-MEMORY BM25 ALGORITHM
# ==========================================================
class BM25:
    """
    Pure Python BM25 indexer.
    """
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(d) for d in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs = []
        self.doc_lens = []
        self.df = Counter()
        
        for doc in corpus:
            self.doc_freqs.append(Counter(doc))
            self.doc_lens.append(len(doc))
            self.df.update(set(doc))
            
        self.idf = {}
        for word, freq in self.df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: List[str], index: int) -> float:
        score = 0.0
        doc_len = self.doc_lens[index]
        freqs = self.doc_freqs[index]
        for word in query:
            if word in freqs:
                f = freqs[word]
                idf = self.idf.get(word, 0.0)
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
        return score

# ==========================================================
# HYBRID RETRIEVER
# ==========================================================
RAG_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "who", "what", "where", "how", "when", "why", "which",
    "of", "for", "in", "on", "at", "by", "to", "with", "from", "about",
    "tell", "me", "give", "show", "do", "does", "did", "can", "could", "would", "should",
    "and", "or", "but", "if", "your", "my", "our", "their", "this", "that", "it", "they"
}

class Retriever:
    """
    Advanced Hybrid Retriever (BM25 + TF-IDF Vector + Intent Filtering).
    """
    def __init__(self, min_score: float = 0.05):
        self.min_score = min_score
        self.bm25 = None
        self.indexed_doc_ids = []
        _store.init()

    def clean_query(self, query: str) -> List[str]:
        if not query:
            return []
        query = query.lower()
        query = re.sub(r"[^a-z0-9\s]", " ", query)
        tokens = re.sub(r"\s+", " ", query).strip().split()
        meaningful = [t for t in tokens if t not in RAG_STOP_WORDS]
        return meaningful if meaningful else tokens

    def _ensure_bm25_initialized(self):
        """Builds BM25 index on documents loaded in _store."""
        _store.build_index()
        docs = _store.documents
        if not docs:
            return
            
        # Rebuild only if document count changed
        doc_signatures = [d["text"] for d in docs]
        if self.bm25 is None or len(self.indexed_doc_ids) != len(docs):
            tokenized_corpus = [self.clean_query(d["text"]) for d in docs]
            self.bm25 = BM25(tokenized_corpus)
            self.indexed_doc_ids = doc_signatures
            logger.info(f"[Retriever] Re-initialized BM25 index with {len(docs)} documents.")

    def retrieve(self, query: str, intent: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Retrieves top documents using BM25 and Vector Search, boosted by intent match.
        """
        self._ensure_bm25_initialized()
        if not _store.documents:
            return []

        # 1. Get Vector Cosine similarities
        # Get extra candidate pool for hybrid merger
        vector_docs = _store.retrieve(query, top_k=max(top_k * 3, 15))
        vector_scores = {vd["text"]: vd["score"] for vd in vector_docs}

        # 2. Get BM25 scores
        query_tokens = self.clean_query(query)
        bm25_scores = {}
        if self.bm25 is not None:
            for idx, doc in enumerate(_store.documents):
                bm25_scores[doc["text"]] = self.bm25.score(query_tokens, idx)

        # Normalize BM25 scores to [0, 1] range for fair merging
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0
        if max_bm25 > 0:
            for k in bm25_scores:
                bm25_scores[k] /= max_bm25

        # 3. Hybrid merger with Intent boost
        hybrid_results = []
        seen_texts = set()
        
        for doc in _store.documents:
            text = doc["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)

            v_score = vector_scores.get(text, 0.0)
            b_score = bm25_scores.get(text, 0.0)
            
            # Combine scores: 50% Vector similarity, 50% BM25 keyword score
            combined_score = 0.5 * v_score + 0.5 * b_score
            
            # Apply intent classifier filtering / boosting
            # If doc intent matches query intent, boost score to prioritize it
            intent_match = False
            if intent and doc.get("intent") == intent:
                combined_score += 0.15
                intent_match = True

            if combined_score < self.min_score:
                continue

            hybrid_results.append({
                "text": doc.get("answer", doc["text"]),
                "question": doc.get("question", ""),
                "answer": doc.get("answer", doc["text"]),
                "source": doc.get("source", doc.get("question", "")),
                "intent": doc.get("intent", "general"),
                "score": round(combined_score, 4),
                "vector_score": round(v_score, 4),
                "bm25_score": round(b_score, 4),
                "intent_match": intent_match
            })

        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        
        final_docs = hybrid_results[:top_k]
        logger.info(f"[Retriever] Retrieved {len(final_docs)} hybrid chunks. Query intent: {intent}")
        return final_docs

    def retrieve_context(self, query: str, intent: Optional[str] = None, top_k: int = 5) -> str:
        docs = self.retrieve(query, intent, top_k)
        return "\n\n".join(d["text"] for d in docs)

# Singleton
retriever = Retriever()

__all__ = ["Retriever", "retriever"]
