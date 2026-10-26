"""
backend/app/rag/qdrant.py
----------------------------------------------------
Qdrant Vector Store for Dense Semantic Retrieval in Genkit AI.
Provides dense embedding vector search complementary to lexical BM25/TF-IDF.
Supports both local in-memory mode (no server) and remote Qdrant server mode.
"""

from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logger import logger
from app.rag.chunker import DocumentChunk

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    PointStruct = None  # type: ignore
    VectorParams = None  # type: ignore
    _QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

# Module-level cache to avoid reloading the 90MB sentence transformer
_SHARED_ENCODER: Optional[Any] = None


def get_shared_encoder() -> Optional[Any]:
    """Returns a cached SentenceTransformer instance or loads it once."""
    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: python -m pip install sentence-transformers"
            )
        _SHARED_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _SHARED_ENCODER


class QdrantRetriever:
    """Dense Semantic Retriever powered by Qdrant vector database."""

    COLLECTION_NAME = "genkit_knowledge"
    VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimension

    def __init__(
        self,
        chunks: List[DocumentChunk],
        collection_name: str = COLLECTION_NAME,
    ):
        self.chunks = chunks
        self.doc_count = len(chunks)
        self.id_to_idx: Dict[str, int] = {
            c.id: idx for idx, c in enumerate(chunks)
        }
        self.collection_name = collection_name
        self._client: Optional[Any] = None
        self._encoder: Optional[Any] = None
        self._available: bool = False

        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Initialises Qdrant client and sentence encoder, syncs collection."""
        if not _QDRANT_AVAILABLE or QdrantClient is None:
            logger.warning(
                "qdrant-client not installed. Dense vector search disabled. "
                "Install with: python -m pip install qdrant-client "
                "sentence-transformers"
            )
            return

        try:
            self._encoder = get_shared_encoder()
        except Exception as e:
            logger.warning(
                f"Failed to load SentenceTransformer: {e}. "
                "Dense vector search disabled."
            )
            return

        try:
            qdrant_url = getattr(settings, "QDRANT_URL", None)
            qdrant_api_key = getattr(settings, "QDRANT_API_KEY", None)

            if qdrant_url:
                # Remote Qdrant server (Docker / Qdrant Cloud)
                self._client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key or None,
                )
                logger.info(
                    f"QdrantRetriever: Connected to remote Qdrant at "
                    f"{qdrant_url}"
                )
            else:
                # Local in-memory Qdrant (no server required - ideal for dev)
                self._client = QdrantClient(":memory:")
                logger.info(
                    "QdrantRetriever: Using in-memory Qdrant "
                    "(no server required)."
                )

            # Ensure collection exists with correct config
            self._ensure_collection()
            self._sync_index()
            self._available = True
            logger.info(
                f"QdrantRetriever ready - collection '{self.collection_name}' "
                f"with {self.doc_count} vectors."
            )

        except Exception as e:
            logger.error(f"QdrantRetriever initialization failed: {e}")
            self._client = None
            self._available = False

    def _ensure_collection(self) -> None:
        """Creates the Qdrant collection if it does not already exist."""
        if self._client is None or VectorParams is None or Distance is None:
            return

        existing = [
            c.name for c in self._client.get_collections().collections
        ]
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                f"QdrantRetriever: Created collection "
                f"'{self.collection_name}'."
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _sync_index(self) -> None:
        """Encodes all DocumentChunks and upserts them into Qdrant."""
        if (
            not self._client
            or not self._encoder
            or not self.chunks
            or PointStruct is None
        ):
            return

        try:
            texts = [f"{c.title}\n{c.text}" for c in self.chunks]
            vectors = self._encoder.encode(
                texts, batch_size=64, show_progress_bar=False
            ).tolist()

            points = [
                PointStruct(
                    id=idx,
                    vector=vectors[idx],
                    payload={
                        "chunk_id": c.id,
                        "source": str(c.source),
                        "category": str(c.category),
                        "title": str(c.title),
                        "priority": int(c.priority),
                    },
                )
                for idx, c in enumerate(self.chunks)
            ]

            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points[i: i + batch_size],
                )

        except Exception as e:
            logger.error(f"QdrantRetriever: Failed to sync index: {e}")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def score_query(self, query: str) -> List[float]:
        """
        Computes cosine similarity scores for all document chunks for a query.
        Returns a float list aligned with self.chunks order (index -> score).
        """
        scores = [0.0] * self.doc_count
        if not self._available or not self._client or not self._encoder:
            return scores
        if not query.strip() or self.doc_count == 0:
            return scores

        try:
            query_vector = self._encoder.encode(
                [query], show_progress_bar=False
            )[0].tolist()

            n_results = min(self.doc_count, 40)

            # Support modern qdrant-client (>=1.9 query_points) & legacy search
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=n_results,
                    with_payload=False,
                )
                hits = response.points
            elif hasattr(self._client, "search"):
                hits = self._client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=n_results,
                    with_payload=False,
                )
            else:
                hits = []

            for hit in hits:
                # hit.id == the integer idx we used during upsert
                idx = int(hit.id)
                if 0 <= idx < self.doc_count:
                    # Qdrant cosine returns score in [-1, 1]; clamp to [0, 1]
                    scores[idx] = max(0.0, float(hit.score))

        except Exception as e:
            logger.warning(
                f"QdrantRetriever.score_query failed: {e}. "
                "Returning zero scores."
            )
        return scores

    @property
    def is_available(self) -> bool:
        """True if Qdrant is initialised and ready for queries."""
        return self._available
