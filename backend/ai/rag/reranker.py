"""
reranker.py
------------------------------------------------------
Genkit AI - Advanced RAG Re-Ranker
Ranks retrieved documents before sending them
to the custom GPT model.
Features
--------
- Keyword overlap scoring
- Exact phrase matching
- Query coverage
- Similarity score weighting
- Length normalization
- Stable ranking
- No external dependencies
Author : Genkit AI
"""
import re
from typing import List, Dict

class Reranker:
    def __init__(self):
        self.stop_words = {
            "a","an","the","is","am","are","was","were",
            "be","been","being","to","of","for","and",
            "or","but","if","then","this","that","these",
            "those","in","on","at","by","with","from",
            "about","into","over","after","before",
            "can","could","would","should","will","shall",
            "may","might","do","does","did","have",
            "has","had"
        }
    # --------------------------------------------------
    def clean(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    # --------------------------------------------------
    def keywords(self, text: str):
        words = self.clean(text).split()
        return {
            w
            for w in words
            if w not in self.stop_words
        }
    # --------------------------------------------------
    def keyword_overlap_score(
        self,
        query_words,
        document_words,
    ):
        if not query_words:
            return 0.0
        overlap = len(
            query_words & document_words
        )
        return overlap / len(query_words)
    # --------------------------------------------------
    def phrase_match_score(
        self,
        query,
        document,
    ):
        q = self.clean(query)
        d = self.clean(document)
        if q in d:
            return 1.0
        return 0.0
    # --------------------------------------------------
    def coverage_score(
        self,
        query_words,
        document_words,
    ):
        if not document_words:
            return 0.0
        overlap = len(
            query_words & document_words
        )
        return overlap / len(document_words)
    # --------------------------------------------------
    def score(
        self,
        query: str,
        document: Dict,
    ) -> float:
        text = document.get("text", "")
        similarity = float(
            document.get("score", 0.0)
        )
        q_words = self.keywords(query)
        d_words = self.keywords(text)
        overlap = self.keyword_overlap_score(
            q_words,
            d_words,
        )
        phrase = self.phrase_match_score(
            query,
            text,
        )
        coverage = self.coverage_score(
            q_words,
            d_words,
        )
        length_bonus = min(
            len(d_words) / 100,
            0.20,
        )
        final_score = (
            similarity * 0.55 +
            overlap * 0.25 +
            coverage * 0.10 +
            phrase * 0.10 +
            length_bonus
        )
        return round(final_score, 4)
    # --------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        if not documents:
            return []
        ranked = []
        seen = set()
        for doc in documents:
            text = doc.get("text", "")
            if text in seen:
                continue
            seen.add(text)
            item = dict(doc)
            item["rerank_score"] = self.score(
                query,
                item,
            )
            ranked.append(item)
        ranked.sort(
            key=lambda x: (
                x["rerank_score"],
                x.get("score", 0),
            ),
            reverse=True,
        )
        return ranked[:top_k]

# ======================================================
# Global Singleton
# ======================================================
reranker = Reranker()
