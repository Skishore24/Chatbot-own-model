"""
backend/app/rag/pipeline.py
----------------------------------------------------
Unified Hybrid RAG Pipeline for Genkit AI V6.1.
End-to-end execution:
Dataset Ingestion -> Inverted Index -> BM25 + TF-IDF -> Hybrid Reranker -> Grounding Validator -> Dynamic Knowledge Synthesis.
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
    def validator(self) -> GroundingValidator:
        """Alias for grounding validator."""
        return self.grounding

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

        # Run BM25 & TF-IDF scoring
        bm25_scores = self.bm25.score_query(query)
        tfidf_scores = self.tfidf.score_query(query)

        # Rerank and extract top-K candidate chunks
        ranked_chunks = self.reranker.rerank(query, self.chunks, bm25_scores, tfidf_scores, top_k=k)

        # Compute grounding confidence and domain verification
        top_score = ranked_chunks[0].score if ranked_chunks else 0.0
        confidence, is_grounded = self.grounding.compute_grounding_score(query, ranked_chunks, top_retrieval_score=top_score)

        if not is_grounded:
            return [], confidence, False

        return ranked_chunks, confidence, is_grounded

    def synthesize_answer(self, query: str, chunks: List[DocumentChunk]) -> str:
        """
        Dynamically synthesizes a clean, authoritative Markdown answer
        from retrieved verified Genkit knowledge chunks.
        """
        if not chunks:
            return self.get_refusal_answer()

        # Handle simple greetings if query is purely conversational greeting
        q_clean = query.strip().lower()
        if q_clean in {"hi", "hello", "hey", "good morning", "good evening", "greetings"}:
            return (
                "Hello! I am the **Genkit AI Assistant**.\n\n"
                "I can help you explore our services, project pricing, technology stack, portfolio, "
                "team, and company policies. How can I assist you today?"
            )

        # Synthesize dynamically from top retrieved chunks
        top_chunk = chunks[0]

        # If the primary top chunk has high relevance and contains comprehensive text
        if len(chunks) == 1 or top_chunk.score > 0.65:
            text = top_chunk.text.strip()
            # Clean Q: prefix if formatted as FAQ
            if text.startswith("**Q:") or text.startswith("Q:"):
                lines = text.split("\n", 1)
                if len(lines) > 1:
                    text = lines[1].strip()
            return text

        # Multi-chunk synthesis: combine top relevant knowledge items cleanly
        synthesized_parts = []
        seen_texts = set()

        for chunk in chunks[:3]:
            chunk_body = chunk.text.strip()
            if chunk_body in seen_texts:
                continue
            seen_texts.add(chunk_body)

            # Strip leading Q: if present
            if chunk_body.startswith("**Q:") or chunk_body.startswith("Q:"):
                lines = chunk_body.split("\n", 1)
                if len(lines) > 1:
                    chunk_body = lines[1].strip()

            synthesized_parts.append(chunk_body)

        if not synthesized_parts:
            return top_chunk.text.strip()

        return "\n\n---\n\n".join(synthesized_parts)

    def build_prompt(self, query: str, chunks: List[DocumentChunk]) -> str:
        """
        Builds structured internal prompt with system instructions and retrieved context for LLM generation.
        """
        context_text = "\n\n".join([f"[{c.title}]\n{c.text}" for c in chunks]) if chunks else "No relevant context found."

        prompt = (
            "SYSTEM:\n"
            "You are Genkit AI, an enterprise AI assistant for Genkit.in.\n"
            "Use only verified knowledge in CONTEXT to answer the QUESTION.\n"
            "Be concise, professional, and directly helpful.\n\n"
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
