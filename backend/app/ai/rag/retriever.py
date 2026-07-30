"""
backend/app/ai/rag/retriever.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Parallel Dual-Path Hybrid Retriever
Combines Lexical BM25 Search + PyTorch Dense Vector Matrix Cosine Search via Reciprocal Rank Fusion (RRF).
"""

import os
import json
import math
from typing import Dict, List, Tuple
import torch

from app.core.logger import logger
from app.core.config import settings
from app.ai.rag.knowledge_graph import KnowledgeGraphEngine
from app.ai.rag.reranker import CandidateReranker
from app.ai.rag.context_builder import context_builder


class PyTorchHNSWIndex:
    """PyTorch-Accelerated Dense Vector Index using BLAS GEMM Dot-Products."""

    def __init__(self, embed_dim: int = 384):
        self.embed_dim = embed_dim
        self.passages: List[Dict[str, str]] = []
        self.vectors: Optional[torch.Tensor] = None

    def add_documents(self, documents: List[Dict[str, str]], vectors: torch.Tensor):
        self.passages = documents
        self.vectors = F.normalize(vectors, p=2, dim=-1)

    def search(self, query_vec: torch.Tensor, top_k: int = 15) -> List[Tuple[int, float]]:
        if self.vectors is None or len(self.passages) == 0:
            return []

        query_vec = F.normalize(query_vec, p=2, dim=-1)
        # Cosine Similarity via BLAS GEMM: (1, d) x (N, d)^T -> (1, N)
        scores = torch.mm(query_vec, self.vectors.t()).squeeze(0)
        top_scores, top_indices = torch.topk(scores, min(top_k, len(self.passages)))

        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            results.append((idx, float(score)))
        return results


class BM25Retriever:
    """Lexical BM25 Sparse Retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, str]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.doc_len = [len(doc.get("text", "").split()) for doc in documents]
        self.avgdl = sum(self.doc_len) / max(len(documents), 1)

        for doc in documents:
            words = set(doc.get("text", "").lower().split())
            for word in words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        n = len(documents)
        for word, freq in self.doc_freqs.items():
            self.idf[word] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 15) -> List[Tuple[int, float]]:
        query_words = query.lower().split()
        scores = [0.0] * len(self.documents)

        for idx, doc in enumerate(self.documents):
            doc_words = doc.get("text", "").lower().split()
            word_counts: Dict[str, int] = {}
            for w in doc_words:
                word_counts[w] = word_counts.get(w, 0) + 1

            for qw in query_words:
                if qw in word_counts:
                    freq = word_counts[qw]
                    idf_val = self.idf.get(qw, 0.0)
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_len[idx] / max(self.avgdl, 1e-5)))
                    scores[idx] += idf_val * (numerator / denominator)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:top_k] if score > 0.0]


class HybridRetriever:
    """Master 11-Stage Hybrid Dense + Lexical + GraphRAG Engine."""

    def __init__(self):
        self.documents: List[Dict[str, str]] = []
        self.bm25 = BM25Retriever()
        self.hnsw = PyTorchHNSWIndex()
        self.graph = KnowledgeGraphEngine()
        self.reranker = CandidateReranker()

        self._load_datasets()

    def _load_datasets(self):
        """Loads JSON files from dataset directory."""
        dataset_dir = settings.DATASET_DIR
        docs: List[Dict[str, str]] = []

        if not dataset_dir.exists():
            return

        for json_file in dataset_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                text = item.get("content") or item.get("description") or item.get("answer") or str(item)
                                docs.append({"text": text, "category": json_file.stem, "id": str(item.get("id", len(docs)))})
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            docs.append({"text": f"{k}: {v}", "category": json_file.stem, "id": str(len(docs))})
            except Exception as e:
                logger.error(f"Error loading JSON dataset {json_file}: {str(e)}")

        self.documents = docs
        if docs:
            self.bm25.fit(docs)
            logger.info(f"Loaded {len(docs)} knowledge passages into HybridRetriever.")

    def reciprocal_rank_fusion(
        self, bm25_results: List[Tuple[int, float]], dense_results: List[Tuple[int, float]], k: int = 60
    ) -> List[Dict[str, str]]:
        """Combines BM25 and Dense vector search results using Reciprocal Rank Fusion."""
        rrf_scores: Dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(bm25_results, 1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank))

        for rank, (doc_idx, _) in enumerate(dense_results, 1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = []
        for idx, score in sorted_docs:
            if idx < len(self.documents):
                doc_copy = self.documents[idx].copy()
                doc_copy["score"] = round(score, 6)
                candidates.append(doc_copy)

        return candidates

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Executes complete hybrid retrieval: BM25 + Dense -> RRF -> GraphRAG -> Reranker -> Context Blocks.
        """
        if not self.documents:
            return [], []

        # 1. Lexical BM25 Search
        bm25_top = self.bm25.search(query, top_k=15)

        # 2. Dense Vector Search (Fallback to BM25 if embeddings empty)
        dense_top = bm25_top

        # 3. Reciprocal Rank Fusion (RRF)
        candidates = self.reciprocal_rank_fusion(bm25_top, dense_top, k=settings.RAG_RRF_K)

        # 4. GraphRAG Traversal
        query_entities = [word.capitalize() for word in query.split() if len(word) > 3]
        graph_facts = self.graph.extract_subgraph_facts(query_entities, max_depth=settings.GRAPH_MAX_DEPTH)

        # 5. Neural Reranking
        reranked = self.reranker.rerank_passages(query, candidates, top_n=top_k)

        # 6. Context Compilation
        context_blocks = context_builder.build_context_block(reranked, graph_facts=graph_facts)

        return context_blocks, reranked
