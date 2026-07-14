"""
ai/rag/retriever.py
----------------------------------------------------
Genkit AI - Hybrid Retriever
Retrieves the most relevant documents from the
Vector Store before reranking.
Features
--------
- TF-IDF retrieval
- Query normalization
- Duplicate removal
- Similarity filtering
- Metadata support
- Ready for future embedding retrieval
Author : Genkit AI
"""
import os
import re
import sys
from typing import List, Dict
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    ),
)
from ai.embeddings.embedding import _store
from config import logger

class Retriever:
    def __init__(
        self,
        min_score: float = 0.05,
    ):
        self.min_score = min_score
        _store.init()
    # --------------------------------------------------
    def clean_query(
        self,
        query: str,
    ) -> str:
        if not query:
            return ""
        query = query.lower()
        query = re.sub(
            r"[^a-z0-9\s]",
            " ",
            query,
        )
        query = re.sub(
            r"\s+",
            " ",
            query,
        )
        return query.strip()
    # --------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        query = self.clean_query(query)
        raw_docs = _store.retrieve(
            query,
            top_k=max(top_k * 2, 10),
        )
        if not raw_docs:
            logger.info(
                "[Retriever] No documents found."
            )
            return []
        results = []
        seen = set()
        for rank, doc in enumerate(raw_docs):
            if doc in seen:
                continue
            seen.add(doc)
            if "\n" in doc:
                question, answer = doc.split(
                    "\n",
                    1,
                )
            else:
                question = "Genkit Knowledge Base"
                answer = doc
            score = max(
                0.0,
                1.0 - (rank * 0.05),
            )
            if score < self.min_score:
                continue
            results.append({
                "text": answer.strip(),
                "source": question.strip(),
                "score": round(score, 4),
                "rank": rank + 1,
            })
            if len(results) >= top_k:
                break
        logger.info(
            f"[Retriever] Retrieved {len(results)} document(s)"
        )
        return results
    # --------------------------------------------------
    def retrieve_texts(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        docs = self.retrieve(
            query,
            top_k,
        )
        return [
            d["text"]
            for d in docs
        ]
    # --------------------------------------------------
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        docs = self.retrieve(
            query,
            top_k,
        )
        return "\n\n".join(
            d["text"]
            for d in docs
        )
    # --------------------------------------------------
    def exists(
        self,
        query: str,
    ) -> bool:
        return len(
            self.retrieve(
                query,
                top_k=1,
            )
        ) > 0

# ==========================================================
# Global Singleton
# ==========================================================
retriever = Retriever()
