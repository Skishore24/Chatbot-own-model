"""
backend/app/rag/pipeline.py
----------------------------------------------------
Unified Hybrid RAG Pipeline for Genkit AI V6.
End-to-end execution:
Dataset Ingestion -> Inverted Index -> BM25 + TF-IDF -> Hybrid Reranker -> Grounding Validator -> Structured Prompt.
"""

from typing import List, Optional, Tuple
from app.core.config import settings
from app.core.logger import logger
from app.rag.chunker import DocumentChunk
from app.rag.loader import load_domain_chunks
from app.rag.index import InvertedIndex
from app.rag.bm25 import BM25Retriever
from app.rag.tfidf import TFIDFRetriever
from app.rag.reranker import HybridReranker
from app.rag.grounding import GroundingValidator, DOMAIN_REFUSAL_MESSAGE


class HybridRAGPipeline:
    """Production Hybrid RAG Pipeline using 100% deterministic local algorithms."""

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None):
        self.chunks: List[DocumentChunk] = chunks if chunks is not None else load_domain_chunks()
        corpus_texts = [f"{c.title} {c.text}" for c in self.chunks]

        self.index = InvertedIndex(corpus_texts)
        self.bm25 = BM25Retriever(self.index, k1=settings.RAG_BM25_K1, b=settings.RAG_BM25_B)
        self.tfidf = TFIDFRetriever(self.index)
        self.reranker = HybridReranker(
            bm25_weight=settings.RAG_FUSION_BM25_WEIGHT,
            tfidf_weight=settings.RAG_FUSION_TFIDF_WEIGHT,
        )
        self.grounding = GroundingValidator(confidence_threshold=settings.RAG_CONFIDENCE_THRESHOLD)

        logger.info(f"Initialized HybridRAGPipeline with {len(self.chunks)} knowledge chunks.")

    @property
    def total_documents(self) -> int:
        return len(self.chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[DocumentChunk], float, bool]:
        """
        Executes hybrid retrieval, reranking, and grounding validation.
        Returns: (top_chunks, confidence_score, is_grounded)
        """
        k = top_k or settings.RAG_TOP_K
        if not self.chunks or not query.strip():
            return [], 0.0, False

        # Check out-of-domain rejection upfront
        if not self.grounding.is_in_domain(query):
            return [], 0.0, False

        # Run BM25 & TF-IDF scoring
        bm25_scores = self.bm25.score_query(query)
        tfidf_scores = self.tfidf.score_query(query)

        # Rerank and extract top-K chunks
        ranked_chunks = self.reranker.rerank(query, self.chunks, bm25_scores, tfidf_scores, top_k=k)

        # Grounding check
        confidence, is_grounded = self.grounding.compute_grounding_score(query, ranked_chunks)

        return ranked_chunks, confidence, is_grounded

    def build_prompt(self, query: str, chunks: List[DocumentChunk]) -> str:
        """
        Builds internal structured prompt with system instructions and retrieved context.
        """
        context_text = "\n\n".join([f"[{c.title}]\n{c.text}" for c in chunks]) if chunks else "No relevant context found."

        prompt = (
            "SYSTEM:\n"
            "You are Genkit AI, an enterprise AI assistant for Genkit.in.\n"
            "RULES:\n"
            "- Answer using only the verified Genkit knowledge provided in CONTEXT.\n"
            "- If the question cannot be answered from the context, state that you don't have verified info.\n"
            "- Do not invent company facts, pricing, or team members.\n"
            "- Be concise, professional, and helpful.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{query}\n\n"
            "ANSWER:\n"
        )
        return prompt

    def get_refusal_answer(self) -> str:
        return DOMAIN_REFUSAL_MESSAGE


# Singleton Pipeline
default_rag_pipeline: Optional[HybridRAGPipeline] = None


def get_rag_pipeline() -> HybridRAGPipeline:
    global default_rag_pipeline
    if default_rag_pipeline is None:
        default_rag_pipeline = HybridRAGPipeline()
    return default_rag_pipeline
