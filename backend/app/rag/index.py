"""
backend/app/rag/index.py
----------------------------------------------------
Inverted Index and Text Normalizer for Lexical Retrieval.
"""

import re
from typing import Dict, List, Set


def normalize_tokens(text: str) -> List[str]:
    """Tokenizes and normalizes text into lowercase alphanumeric stems/tokens."""
    if not text:
        return []
    # Tokenize words, numbers, and tech terms (e.g. ui/ux, next.js)
    tokens = re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text.lower())
    return [t.strip(".-") for t in tokens if len(t.strip(".-")) > 1]


class InvertedIndex:
    """Inverted index mapping terms to chunk indices and term frequencies."""

    def __init__(self, corpus: List[str]):
        self.doc_count = len(corpus)
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.postings: Dict[str, Dict[int, int]] = {}  # term -> {doc_idx: freq}
        self.df: Dict[str, int] = {}  # term -> document frequency

        self._build_index(corpus)

    def _build_index(self, corpus: List[str]) -> None:
        total_length = 0
        for doc_id, doc in enumerate(corpus):
            tokens = normalize_tokens(doc)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            seen_in_doc: Set[str] = set()
            for token in tokens:
                if token not in self.postings:
                    self.postings[token] = {}
                self.postings[token][doc_id] = self.postings[token].get(doc_id, 0) + 1

                if token not in seen_in_doc:
                    self.df[token] = self.df.get(token, 0) + 1
                    seen_in_doc.add(token)

        self.avg_doc_len = total_length / max(self.doc_count, 1)
