"""
backend/app/ai/rag/retriever.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Parallel Dual-Path Hybrid Retriever
Pipeline:
  1. Query Expansion  (synonym-based term boosting)
  2. BM25 Sparse Search
  3. TF-IDF Dense Vector Cosine Search (PyTorch BLAS GEMM)
  4. Reciprocal Rank Fusion (RRF)
  5. GraphRAG BFS Traversal
  6. Neural Reranking  (n-gram overlap + density scoring)
  7. MMR Diversification  (Maximal Marginal Relevance)
  8. Context Compilation (max_chars budget)
100% local, zero external APIs.
"""

import json
import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from app.core.logger import logger
from app.core.config import settings
from app.ai.rag.knowledge_graph import KnowledgeGraphEngine
from app.ai.rag.reranker import CandidateReranker
from app.ai.rag.context_builder import context_builder
from app.ai.rag.embedder import TFIDFEmbedder


_SYNONYM_MAP: Dict[str, List[str]] = {
    "price": ["cost", "pricing", "rate", "fee", "budget", "plan"],
    "service": ["offering", "solution", "product", "package"],
    "ai": ["artificial intelligence", "machine learning", "llm", "model"],
    "web": ["website", "frontend", "react", "nextjs"],
    "app": ["application", "mobile", "flutter", "android", "ios"],
    "team": ["staff", "developer", "engineer", "expert"],
    "contact": ["email", "phone", "reach", "get in touch"],
    "portfolio": ["project", "work", "case study", "client"],
    "technology": ["tech stack", "framework", "tool", "language"],
}


def _expand_query(query: str) -> str:
    expanded_terms: List[str] = []
    q_lower = query.lower()
    for seed, synonyms in _SYNONYM_MAP.items():
        if seed in q_lower:
            expanded_terms.extend(synonyms)
    if expanded_terms:
        return query + " " + " ".join(expanded_terms)
    return query


class PyTorchHNSWIndex:
    def __init__(self, embed_dim: int = 8192):
        self.embed_dim = embed_dim
        self.passages: List[Dict[str, str]] = []
        self.vectors: Optional[torch.Tensor] = None

    def add_documents(self, documents: List[Dict[str, str]], vectors: torch.Tensor) -> None:
        self.passages = documents
        self.vectors = F.normalize(vectors.float(), p=2, dim=-1)
        logger.info(f"PyTorchHNSWIndex: {len(documents)} vectors loaded. Shape: {self.vectors.shape}")

    def search(self, query_vec: torch.Tensor, top_k: int = 15) -> List[Tuple[int, float]]:
        if self.vectors is None or len(self.passages) == 0:
            return []
        query_vec = F.normalize(query_vec.float(), p=2, dim=-1)
        scores = torch.mm(query_vec, self.vectors.t()).squeeze(0)
        top_k_actual = min(top_k, len(self.passages))
        top_scores, top_indices = torch.topk(scores, top_k_actual)
        return [(int(idx), float(score)) for idx, score in zip(top_indices.tolist(), top_scores.tolist())]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, str]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[Dict[str, str]]) -> None:
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
                    num = freq * (self.k1 + 1.0)
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_len[idx] / max(self.avgdl, 1e-5)))
                    scores[idx] += idf_val * (num / denom)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:top_k] if score > 0.0]


def _maximal_marginal_relevance(
    candidates: List[Dict[str, str]],
    embedder: TFIDFEmbedder,
    top_k: int = 5,
    lambda_param: float = 0.6,
) -> List[Dict[str, str]]:
    if not candidates or not embedder.is_fitted:
        return candidates[:top_k]
    texts = [c.get("text", "") for c in candidates]
    vecs = F.normalize(embedder.encode_batch(texts).float(), p=2, dim=-1)
    selected_indices: List[int] = []
    remaining_indices = list(range(len(candidates)))
    while len(selected_indices) < top_k and remaining_indices:
        if not selected_indices:
            best_idx = max(remaining_indices, key=lambda i: candidates[i].get("rerank_score", candidates[i].get("score", 0.0)))
        else:
            selected_vecs = vecs[selected_indices]
            best_idx = None
            best_mmr = -float("inf")
            for i in remaining_indices:
                relevance = candidates[i].get("rerank_score", candidates[i].get("score", 0.0))
                sim = torch.mm(vecs[i].unsqueeze(0), selected_vecs.t()).max().item()
                mmr = lambda_param * relevance - (1.0 - lambda_param) * sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    return [candidates[i] for i in selected_indices]


class HybridRetriever:
    def __init__(self):
        self.documents: List[Dict[str, str]] = []
        self.bm25 = BM25Retriever()
        self.hnsw = PyTorchHNSWIndex()
        self.embedder = TFIDFEmbedder()
        self.graph = KnowledgeGraphEngine()
        self.reranker = CandidateReranker()
        self._load_datasets()

    def _load_datasets(self) -> None:
        dataset_dir = settings.DATASET_DIR
        docs: List[Dict[str, str]] = []
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return
        json_files = list(dataset_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files in {dataset_dir}")
            return
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            text = (item.get("content") or item.get("description") or
                                    item.get("answer") or item.get("text") or
                                    item.get("output") or item.get("instruction") or str(item))
                            if text and len(str(text).strip()) > 10:
                                docs.append({"text": str(text).strip(), "category": json_file.stem, "id": str(item.get("id", len(docs)))})
                elif isinstance(data, dict):
                    for k, v in data.items():
                        text = f"{k}: {v}" if not isinstance(v, dict) else str(v)
                        docs.append({"text": text.strip(), "category": json_file.stem, "id": str(len(docs))})
                logger.info(f"Loaded {json_file.name} — total: {len(docs)}")
            except Exception as e:
                logger.error(f"Error loading {json_file.name}: {e}")
        if not docs:
            logger.warning("No documents loaded.")
            return
        self.documents = docs
        self.bm25.fit(docs)
        corpus_texts = [d["text"] for d in docs]
        self.embedder.fit(corpus_texts)
        doc_vectors = self.embedder.encode_batch(corpus_texts)
        self.hnsw = PyTorchHNSWIndex(embed_dim=self.embedder.embedding_dim)
        self.hnsw.add_documents(docs, doc_vectors)
        logger.info(f"HybridRetriever ready — {len(docs)} docs, TF-IDF dim: {self.embedder.embedding_dim:,}")

    def _reciprocal_rank_fusion(self, bm25_results: List[Tuple[int, float]], dense_results: List[Tuple[int, float]], k: int = 60) -> List[Dict[str, str]]:
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
        if not self.documents:
            logger.warning("HybridRetriever has no documents.")
            return [], []
        effective_k = max(top_k, settings.RAG_TOP_K)
        expanded_query = _expand_query(query)
        bm25_top = self.bm25.search(expanded_query, top_k=15)
        if self.embedder.is_fitted:
            query_vec = self.embedder.query_vector(expanded_query)
            dense_top = self.hnsw.search(query_vec, top_k=15)
        else:
            dense_top = bm25_top
        candidates = self._reciprocal_rank_fusion(bm25_top, dense_top, k=settings.RAG_RRF_K)
        query_entities = [w.capitalize() for w in query.split() if len(w) > 3]
        graph_facts = self.graph.extract_subgraph_facts(query_entities, max_depth=settings.GRAPH_MAX_DEPTH)
        reranked = self.reranker.rerank_passages(query, candidates, top_n=effective_k * 2)
        diverse = _maximal_marginal_relevance(reranked, self.embedder, top_k=effective_k, lambda_param=0.6)
        context_blocks = context_builder.build_context_block(diverse, graph_facts=graph_facts)
        logger.info(f"Retrieval: '{query[:50]}' | BM25={len(bm25_top)} Dense={len(dense_top)} RRF={len(candidates)} MMR={len(diverse)} Graph={len(graph_facts)}")
        return context_blocks, diverse
