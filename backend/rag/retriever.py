from typing import List
from services.vector_store import search
from app.config import logger


def retrieve(query: str, top_k: int = 5) -> List[str]:
    """
    Retrieve relevant documents using vector similarity
    """

    try:
        results = search(query, top_k=top_k)

        if not results:
            logger.warning(f"[RAG] No results for query: {query}")
            return []

        # Basic cleanup
        cleaned = []
        for r in results:
            r = r.strip()
            if len(r) > 10:
                cleaned.append(r)

        return cleaned

    except Exception as e:
        logger.error(f"[RAG RETRIEVE ERROR]: {e}")
        return []