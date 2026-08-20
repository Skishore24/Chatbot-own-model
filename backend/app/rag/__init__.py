"""
backend/app/rag/__init__.py
----------------------------------------------------
Public exports for Genkit AI Hybrid RAG module.
"""

from app.rag.chunker import DocumentChunk
from app.rag.loader import load_domain_chunks
from app.rag.index import InvertedIndex, normalize_tokens
from app.rag.bm25 import BM25Retriever
from app.rag.tfidf import TFIDFRetriever
from app.rag.reranker import HybridReranker
from app.rag.grounding import GroundingValidator, DOMAIN_REFUSAL_MESSAGE
from app.rag.pipeline import HybridRAGPipeline, get_rag_pipeline

__all__ = [
    "DocumentChunk",
    "load_domain_chunks",
    "InvertedIndex",
    "normalize_tokens",
    "BM25Retriever",
    "TFIDFRetriever",
    "HybridReranker",
    "GroundingValidator",
    "DOMAIN_REFUSAL_MESSAGE",
    "HybridRAGPipeline",
    "get_rag_pipeline",
]
