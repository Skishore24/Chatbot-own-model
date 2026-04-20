from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import logger

# Lightweight reranker model
_rerank_model = SentenceTransformer("all-MiniLM-L6-v2")


def rerank(query: str, docs: List[str], top_k: int = 3) -> List[str]:
    """
    Re-rank retrieved documents using semantic similarity
    """

    if not docs:
        return []

    try:
        query_vec = _rerank_model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        doc_vecs = _rerank_model.encode(
            docs,
            normalize_embeddings=True
        )

        scores = np.dot(doc_vecs, query_vec)

        ranked = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        return [doc for _, doc in ranked[:top_k]]

    except Exception as e:
        logger.error(f"[RERANK ERROR]: {e}")
        return docs[:top_k]