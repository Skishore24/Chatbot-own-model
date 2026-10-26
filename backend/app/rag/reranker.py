"""
backend/app/rag/reranker.py
----------------------------------------------------
Deterministic Reranker with Reciprocal Rank Fusion (RRF),
Title Match & Term Coverage Boost.
Works with BM25, TF-IDF, and Qdrant dense vector scores.
"""

from typing import List, Optional, Tuple
from app.rag.chunker import DocumentChunk
from app.rag.index import normalize_tokens


class HybridReranker:
    """Combines BM25, TF-IDF, Qdrant dense vectors, and lexical heuristics to rank chunks."""

    def __init__(
        self,
        bm25_weight: float = 0.45,
        tfidf_weight: float = 0.20,
        qdrant_weight: float = 0.35,
        rrf_k: int = 60,
    ):
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight
        self.qdrant_weight = qdrant_weight
        self.rrf_k = rrf_k

    def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        bm25_scores: List[float],
        tfidf_scores: List[float],
        qdrant_scores: Optional[List[float]] = None,
        top_k: int = 4,
    ) -> List[DocumentChunk]:
        """Merges retrieval scores with lexical coverage and returns top-K ranked chunks."""
        if not chunks:
            return []

        query_tokens = set(normalize_tokens(query))
        query_str = query.lower()

        # 1. Normalize BM25 and TFIDF scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        norm_bm25 = [s / max_bm25 for s in bm25_scores]

        max_tfidf = max(tfidf_scores) if max(tfidf_scores) > 0 else 1.0
        norm_tfidf = [s / max_tfidf for s in tfidf_scores]

        # 2. Compute RRF ranks
        bm25_ranks = {
            doc_idx: rank
            for rank, doc_idx in enumerate(
                sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            )
        }
        tfidf_ranks = {
            doc_idx: rank
            for rank, doc_idx in enumerate(
                sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)
            )
        }

        # 3. Process optional Qdrant scores
        has_qdrant = qdrant_scores is not None and any(s > 0 for s in qdrant_scores)
        if has_qdrant:
            max_qdrant = max(qdrant_scores) if max(qdrant_scores) > 0 else 1.0  # type: ignore[arg-type]
            norm_qdrant = [s / max_qdrant for s in qdrant_scores]  # type: ignore[union-attr]
            qdrant_ranks = {
                doc_idx: rank
                for rank, doc_idx in enumerate(
                    sorted(
                        range(len(qdrant_scores)),  # type: ignore[arg-type]
                        key=lambda i: qdrant_scores[i],  # type: ignore[index]
                        reverse=True,
                    )
                )
            }
        else:
            norm_qdrant = [0.0] * len(chunks)
            qdrant_ranks = {}

        # Stopwords to strip before coverage scoring
        COMMON_STOPWORDS = {
            "genkit", "what", "how", "who", "tell", "about", "does", "you",
            "use", "the", "for", "with", "and", "can", "is", "where", "do",
            "we", "our", "your", "me",
        }
        content_tokens = {t for t in query_tokens if t not in COMMON_STOPWORDS}
        eval_tokens = content_tokens if content_tokens else query_tokens

        # Weight normalization when Qdrant is absent
        w_bm25 = self.bm25_weight
        w_tfidf = self.tfidf_weight
        w_qdrant = self.qdrant_weight
        if not has_qdrant:
            total_w = w_bm25 + w_tfidf
            w_bm25 = w_bm25 / max(total_w, 1e-6)
            w_tfidf = w_tfidf / max(total_w, 1e-6)
            w_qdrant = 0.0

        scored_chunks: List[Tuple[float, DocumentChunk]] = []

        for idx, chunk in enumerate(chunks):
            # Reciprocal Rank Fusion component
            rrf_bm25 = 1.0 / (self.rrf_k + bm25_ranks[idx] + 1)
            rrf_tfidf = 1.0 / (self.rrf_k + tfidf_ranks[idx] + 1)
            rrf_qdrant = (
                1.0 / (self.rrf_k + qdrant_ranks.get(idx, len(chunks)) + 1)
                if has_qdrant
                else 0.0
            )

            rrf_score = (
                (w_bm25 * rrf_bm25) + (w_tfidf * rrf_tfidf) + (w_qdrant * rrf_qdrant)
            )

            # Linear normalized fusion
            linear_score = (
                (w_bm25 * norm_bm25[idx])
                + (w_tfidf * norm_tfidf[idx])
                + (w_qdrant * norm_qdrant[idx])
            )

            # Term coverage boost
            chunk_tokens = set(normalize_tokens(chunk.text + " " + chunk.title))
            coverage = len(query_tokens.intersection(chunk_tokens)) / max(
                len(query_tokens), 1
            )
            content_coverage = len(eval_tokens.intersection(chunk_tokens)) / max(
                len(eval_tokens), 1
            )

            # Title & exact phrase boost
            chunk_title_tokens = set(normalize_tokens(chunk.title))
            title_boost = (
                0.40
                if len(eval_tokens.intersection(chunk_title_tokens)) > 0
                else 0.0
            )
            phrase_boost = (
                0.30
                if any(
                    kw in query_str
                    for kw in chunk.keywords
                    if kw not in COMMON_STOPWORDS
                )
                else 0.0
            )

            # Priority multiplier
            priority_mult = 1.0 + (0.10 * (chunk.priority - 1))

            # Final unified score
            final_score = (
                linear_score * 0.30
                + rrf_score * 35.0
                + content_coverage * 0.35
                + coverage * 0.15
                + title_boost
                + phrase_boost
            ) * priority_mult

            chunk.score = float(final_score)
            scored_chunks.append((final_score, chunk))

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
