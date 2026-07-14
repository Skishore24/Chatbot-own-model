"""
backend/ai/embeddings/embedding.py
----------------------------------------------------
Genkit AI - Embedding & Vector Store
Custom TF-IDF implementation.
Features
--------
• No sklearn
• No FAISS
• No ChromaDB
• Own Vector Store
• Fast cosine similarity
• Query caching
• Thread safe
Author : Genkit AI
"""
import json
import math
import re
from collections import Counter
from threading import Lock
from typing import Dict, List
import numpy as np
from config import DATASET_PATH, logger

# ==========================================================
# TF-IDF VECTORIZER
# ==========================================================
class TFIDFVectorizer:
    """
    Production-ready TF-IDF implementation.
    Features
    --------
    • Own tokenizer
    • Own IDF
    • Own TF
    • L2 Normalization
    • Fast cosine search
    """
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0
        self.vocab_size = 0
        self.token_pattern = re.compile(r"\w+")
    # ------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        """
        Convert text into lowercase tokens.
        """
        if not text:
            return []
        return self.token_pattern.findall(text.lower())
    # ------------------------------------------------------
    def clean(self, text: str) -> str:
        """
        Normalize text.
        """
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    # ------------------------------------------------------
    def fit_transform(
        self,
        documents: List[str]
    ) -> np.ndarray:
        self.doc_count = len(documents)
        tokenized_docs = [
            self.tokenize(doc)
            for doc in documents
        ]
        vocabulary = set()
        for doc in tokenized_docs:
            vocabulary.update(doc)
        self.vocab = {
            word: idx
            for idx, word in enumerate(sorted(vocabulary))
        }
        self.vocab_size = len(self.vocab)
        document_frequency = Counter()
        for doc in tokenized_docs:
            document_frequency.update(set(doc))
        self.idf = {}
        for word in self.vocab:
            df = document_frequency[word]
            self.idf[word] = math.log(
                (1 + self.doc_count) /
                (1 + df)
            ) + 1
        return self.vectorize_documents(
            tokenized_docs
        )
    # ------------------------------------------------------
    def transform(
        self,
        documents: List[str]
    ) -> np.ndarray:
        tokenized_docs = [
            self.tokenize(doc)
            for doc in documents
        ]
        return self.vectorize_documents(
            tokenized_docs
        )
    # ------------------------------------------------------
    def transform_query(
        self,
        query: str
    ) -> np.ndarray:
        return self.transform([query])[0]
    # ------------------------------------------------------
    def vectorize_documents(
        self,
        tokenized_docs: List[List[str]]
    ) -> np.ndarray:
        vectors = []
        for words in tokenized_docs:
            vector = np.zeros(
                self.vocab_size,
                dtype=np.float32
            )
            if not words:
                vectors.append(vector)
                continue
            counts = Counter(words)
            total = len(words)
            for word, freq in counts.items():
                if word not in self.vocab:
                    continue
                tf = freq / total
                vector[self.vocab[word]] = (
                    tf *
                    self.idf[word]
                )
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            vectors.append(vector)
        return np.asarray(
            vectors,
            dtype=np.float32
        )
    # ------------------------------------------------------
    @staticmethod
    def cosine_similarity(
        query_vector: np.ndarray,
        document_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Fast cosine similarity.
        """
        return np.dot(
            document_vectors,
            query_vector
        )
    # ==========================================================
# VECTOR STORE
# ==========================================================
class VectorStore:
    """
    Production Vector Store
    Features
    --------
    • TF-IDF Index
    • Hybrid Retrieval
    • Query Cache
    • Thread Safe
    • Session Memory
    • Incremental Document Update
    """
    def __init__(self):
        self.vectorizer = TFIDFVectorizer()
        self.documents: List[dict] = []
        self.embeddings = None
        self._initialized = False
        self._lock = Lock()
        self.memory = {}
        self.cache = {}

    def init(self):
        self.build_index()
    # ------------------------------------------------------
    def clean(self, text: str) -> str:
        return self.vectorizer.clean(text)
    # ------------------------------------------------------
    def load_dataset(self):
        if not DATASET_PATH.exists():
            logger.warning("Dataset not found.")
            return []
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.exception(e)
            return []
        documents = []
        for item in data:
            question = self.clean(
                item.get("instruction", "")
            )
            answer = self.clean(
                item.get("output", "")
            )
            if len(question) < 3:
                continue
            if len(answer) < 3:
                continue
            documents.append({
                "question": question,
                "answer": answer,
                "text": question + "\n" + answer,
                "source": "dataset"
            })
        logger.info(
            f"Loaded {len(documents)} documents."
        )
        return documents
    # ------------------------------------------------------
    def build_index(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self.documents = self.load_dataset()
            if not self.documents:
                logger.warning(
                    "No documents available."
                )
                return
            corpus = [
                doc["text"]
                for doc in self.documents
            ]
            self.embeddings = (
                self.vectorizer.fit_transform(
                    corpus
                )
            )
            self._initialized = True
            logger.info(
                "Vector index created."
            )
    # ------------------------------------------------------
    def retrieve(
        self,
        query,
        top_k=5
    ):
        self.build_index()
        if self.embeddings is None:
            return []
        cache_key = (
            query.lower(),
            top_k
        )
        if cache_key in self.cache:
            return self.cache[cache_key]
        query_vector = (
            self.vectorizer.transform_query(
                query
            )
        )
        similarity = (
            self.vectorizer.cosine_similarity(
                query_vector,
                self.embeddings
            )
        )
        ranked = np.argsort(
            similarity
        )[::-1]
        results = []
        for idx in ranked:
            score = float(
                similarity[idx]
            )
            if score <= 0:
                continue
            document = dict(
                self.documents[idx]
            )
            document["score"] = score
            results.append(
                document
            )
            if len(results) >= top_k:
                break
        self.cache[cache_key] = results
        return results
    # ------------------------------------------------------
    def rerank(
        self,
        query,
        documents,
        top_k=3
    ):
        if not documents:
            return []
        query_words = set(
            self.clean(query).split()
        )
        ranked = []
        for doc in documents:
            doc_words = set(
                doc["text"].split()
            )
            overlap = len(
                query_words & doc_words
            )
            final_score = (
                doc["score"]
                +
                overlap * 0.10
            )
            doc["rerank_score"] = final_score
            ranked.append(doc)
        ranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        return ranked[:top_k]
    # ------------------------------------------------------
    def add_documents(
        self,
        new_documents
    ):
        if not new_documents:
            return
        self.documents.extend(
            new_documents
        )
        corpus = [
            d["text"]
            for d in self.documents
        ]
        self.embeddings = (
            self.vectorizer.fit_transform(
                corpus
            )
        )
        self.cache.clear()
        logger.info(
            "Knowledge base updated."
        )
    # ==========================================================
# SESSION MEMORY
# ==========================================================
    def save_memory(
        self,
        session_id: str,
        text: str
    ):
        with self._lock:
            if session_id not in self.memory:
                self.memory[session_id] = []
            self.memory[session_id].append(text)
            # Keep only last 10 messages
            self.memory[session_id] = (
                self.memory[session_id][-10:]
            )
    # ------------------------------------------------------
    def search_memory(
        self,
        query: str,
        session_id: str
    ) -> str:
        if session_id not in self.memory:
            return ""
        query_words = set(
            self.clean(query).split()
        )
        scored = []
        for memory in self.memory[session_id]:
            memory_words = set(
                self.clean(memory).split()
            )
            overlap = len(
                query_words & memory_words
            )
            if overlap:
                scored.append(
                    (
                        overlap,
                        memory
                    )
                )
        scored.sort(
            reverse=True
        )
        return "\n".join(
            text
            for _, text in scored[:3]
        )
    # ------------------------------------------------------
    def clear_memory(
        self,
        session_id: str
    ):
        with self._lock:
            if session_id in self.memory:
                del self.memory[session_id]
    # ------------------------------------------------------
    def clear_cache(self):
        self.cache.clear()
    # ------------------------------------------------------
    def reset(self):
        with self._lock:
            self.documents.clear()
            self.memory.clear()
            self.cache.clear()
            self.embeddings = None
            self._initialized = False
            logger.info(
                "Vector store reset."
            )

# ==========================================================
# SINGLETON
# ==========================================================
_store = VectorStore()

# ==========================================================
# PUBLIC FUNCTIONS
# ==========================================================
def init_vector_store():
    """
    Initialize vector store.
    """
    _store.build_index()

# ----------------------------------------------------------
def retrieve(
    query: str,
    top_k: int = 5
):
    return _store.retrieve(
        query=query,
        top_k=top_k
    )

# ----------------------------------------------------------
def rerank(
    query: str,
    documents,
    top_k: int = 3
):
    return _store.rerank(
        query=query,
        documents=documents,
        top_k=top_k
    )

# ----------------------------------------------------------
def add_documents(
    documents
):
    _store.add_documents(
        documents
    )

# ----------------------------------------------------------
def save_memory(
    session_id: str,
    text: str
):
    _store.save_memory(
        session_id,
        text
    )

# ----------------------------------------------------------
def search_memory(
    query: str,
    session_id: str
):
    return _store.search_memory(
        query,
        session_id
    )

# ----------------------------------------------------------
def clear_memory(
    session_id: str
):
    _store.clear_memory(
        session_id
    )

# ----------------------------------------------------------
def clear_cache():
    _store.clear_cache()

# ----------------------------------------------------------
def reset_vector_store():
    _store.reset()
