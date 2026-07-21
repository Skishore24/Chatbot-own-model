"""
backend/ai/embeddings/embedding.py
----------------------------------------------------
Genkit AI - Embedding & Vector Store (v4.0)

Custom TF-IDF vector store designed to load from the structured knowledge base files,
supporting hybrid retrieval.

Author : Genkit AI
"""
import os
import json
import math
import re
from collections import Counter
from threading import Lock
from typing import Dict, List, Optional
import numpy as np
from config import DATASET_DIR, logger

# ==========================================================
# TF-IDF VECTORIZER
# ==========================================================
class TFIDFVectorizer:
    """
    Custom TF-IDF implementation.
    """
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0
        self.vocab_size = 0
        self.token_pattern = re.compile(r"\w+")

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return self.token_pattern.findall(text.lower())

    def clean(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        self.doc_count = len(documents)
        tokenized_docs = [self.tokenize(doc) for doc in documents]
        
        vocabulary = set()
        for doc in tokenized_docs:
            vocabulary.update(doc)
            
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocabulary))}
        self.vocab_size = len(self.vocab)
        
        document_frequency = Counter()
        for doc in tokenized_docs:
            document_frequency.update(set(doc))
            
        self.idf = {}
        for word in self.vocab:
            df = document_frequency[word]
            self.idf[word] = math.log((1 + self.doc_count) / (1 + df)) + 1
            
        return self.vectorize_documents(tokenized_docs)

    def transform(self, documents: List[str]) -> np.ndarray:
        tokenized_docs = [self.tokenize(doc) for doc in documents]
        return self.vectorize_documents(tokenized_docs)

    def transform_query(self, query: str) -> np.ndarray:
        return self.transform([query])[0]

    def vectorize_documents(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        vectors = []
        for words in tokenized_docs:
            vector = np.zeros(self.vocab_size, dtype=np.float32)
            if not words:
                vectors.append(vector)
                continue
            counts = Counter(words)
            total = len(words)
            for word, freq in counts.items():
                if word not in self.vocab:
                    continue
                tf = freq / total
                vector[self.vocab[word]] = tf * self.idf[word]
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def cosine_similarity(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
        return np.dot(document_vectors, query_vector)

# ==========================================================
# VECTOR STORE
# ==========================================================
class VectorStore:
    """
    Vector Store mapping structured JSON data to searchable contexts.
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

    def clean(self, text: str) -> str:
        return self.vectorizer.clean(text)

    def load_dataset(self) -> List[dict]:
        """Loads and formats the structured JSON files into doc chunks."""
        if not DATASET_DIR.exists():
            logger.warning("Dataset directory not found.")
            return []
            
        documents = []

        def add_doc(question: str, answer: str, intent: str, source: str):
            q_clean = self.clean(question)
            a_clean = self.clean(answer)
            if len(q_clean) >= 3 and len(a_clean) >= 3:
                documents.append({
                    "question": question,
                    "answer": answer,
                    "text": f"{question}\n{answer}",
                    "intent": intent,
                    "source": source
                })

        # 1. company.json
        p = DATASET_DIR / "company.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    founders = ", ".join(data.get("founders", []))
                    values = ", ".join(data.get("values", []))
                    text = (
                        f"Genkit AI co-founded by {founders} in {data.get('founded')}. "
                        f"Tagline: {data.get('tagline')}. Motto: {data.get('motto')}. "
                        f"Mission: {data.get('mission')}. Vision: {data.get('vision')}. "
                        f"Core values: {values}. Operational setup: {data.get('operational_model')}."
                    )
                    add_doc("Tell me about Genkit company profile vision mission history founders co-founders owner", text, "about", "company")
            except Exception as e:
                logger.error(f"Error loading company.json: {e}")

        # 2. services.json
        p = DATASET_DIR / "services.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        techs = ", ".join(item.get("technologies", []))
                        benefits = ", ".join(item.get("benefits", []))
                        text = (
                            f"{item.get('service_name')} service: {item.get('description')} "
                            f"We utilize technologies: {techs}. Major benefits: {benefits}. "
                            f"Project turnaround: {item.get('turnaround')}."
                        )
                        add_doc(f"What services do you offer for {item.get('service_name')}?", text, "services", "services")
            except Exception as e:
                logger.error(f"Error loading services.json: {e}")

        # 3. projects.json
        p = DATASET_DIR / "projects.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        techs = ", ".join(item.get("technology", []))
                        text = (
                            f"Project: {item.get('project_name')}. Category: {item.get('service')}. "
                            f"Description: {item.get('description')} Stacks used: {techs}. "
                            f"Business impact and results: {item.get('impact')}."
                        )
                        add_doc(f"Tell me about project {item.get('project_name')}", text, "project", "projects")
            except Exception as e:
                logger.error(f"Error loading projects.json: {e}")

        # 4. technologies.json
        p = DATASET_DIR / "technologies.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for category, techs in data.items():
                        for item in techs:
                            text = f"{item.get('name')} is used as a {category} tool/language: {item.get('description')}"
                            add_doc(f"Why do you use {item.get('name')}?", text, "technology", "technologies")
            except Exception as e:
                logger.error(f"Error loading technologies.json: {e}")

        # 5. pricing.json
        p = DATASET_DIR / "pricing.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for pkg in data.get("packages", []):
                        feats = ", ".join(pkg.get("features", []))
                        text = (
                            f"Package {pkg.get('name')} starts at {pkg.get('starting_price')} USD. "
                            f"Features: {feats}. Delivery timeline: {pkg.get('timeline')}."
                        )
                        add_doc(f"How much does {pkg.get('name')} package cost?", text, "pricing", "pricing")
                    rates = ", ".join([f"{k.replace('_',' ')}: {v} USD/hr" for k, v in data.get("hourly_rates", {}).items()])
                    text_rates = f"Our hourly rates are: {rates}. {data.get('flexible_budget')}"
                    add_doc("What are Genkit hourly rates or design budgets?", text_rates, "pricing", "pricing")
            except Exception as e:
                logger.error(f"Error loading pricing.json: {e}")

        # 6. team.json
        p = DATASET_DIR / "team.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        text = (
                            f"{item.get('name')} role is {item.get('role')}. "
                            f"Specialty is {item.get('specialty')}. Background: {item.get('background')}."
                        )
                        add_doc(f"Who is {item.get('name')} at Genkit?", text, "support", "team")
            except Exception as e:
                logger.error(f"Error loading team.json: {e}")

        # 7. clients.json
        p = DATASET_DIR / "clients.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    industries = ", ".join(data.get("industries_served", []))
                    text = f"Genkit serves industries: {industries}."
                    for t in data.get("testimonials", []):
                        text += f" Testimonial from {t.get('client')} ({t.get('company')}): '{t.get('feedback')}'"
                    add_doc("What clients or industries do you serve?", text, "portfolio", "clients")
            except Exception as e:
                logger.error(f"Error loading clients.json: {e}")

        # 8. portfolio.json
        p = DATASET_DIR / "portfolio.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    philosophy = data.get("design_philosophy")
                    cs = data.get("featured_case_study", {})
                    text = (
                        f"Design philosophy: {philosophy}. "
                        f"Featured Case Study: '{cs.get('title')}'. Challenge: {cs.get('challenge')} "
                        f"Solution: {cs.get('solution')} Result: {cs.get('result')}"
                    )
                    add_doc("Show me Genkit case studies and portfolios", text, "portfolio", "portfolio")
            except Exception as e:
                logger.error(f"Error loading portfolio.json: {e}")

        # 9. faq.json
        p = DATASET_DIR / "faq.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        add_doc(item.get("question"), item.get("answer"), "support", "faq")
            except Exception as e:
                logger.error(f"Error loading faq.json: {e}")

        # 10. policies.json
        p = DATASET_DIR / "policies.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = (
                        f"Support Policy: {data.get('support_policy', {}).get('standard')} "
                        f"Maintenance: {data.get('support_policy', {}).get('maintenance_plans')} "
                        f"Revision Rules: {data.get('revision_policy', {}).get('details')} "
                        f"Refund Rules: {data.get('refund_policy', {}).get('details')} "
                        f"NDA and Privacy: {data.get('data_privacy', {}).get('details')}."
                    )
                    add_doc("What are your revision, support, and refund policies?", text, "support", "policies")
            except Exception as e:
                logger.error(f"Error loading policies.json: {e}")

        # 11. contact.json
        p = DATASET_DIR / "contact.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    channels = ", ".join([f"{c.get('type')}: {c.get('address', c.get('handle'))} ({c.get('best_for')})" for c in data.get("channels", [])])
                    text = (
                        f"Genkit email is {data.get('email')}. Website is {data.get('website')}. "
                        f"Contact Form: {data.get('contact_form')}. Active channels: {channels}. "
                        f"Response window: {data.get('response_time')}. Free consultations calendar: {data.get('consultations')}."
                    )
                    add_doc("How to book a call or reach Genkit channels?", text, "contact", "contact")
            except Exception as e:
                logger.error(f"Error loading contact.json: {e}")

        logger.info(f"[VectorStore] Successfully compiled and loaded {len(documents)} structured document chunks.")
        return documents

    def build_index(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self.documents = self.load_dataset()
            if not self.documents:
                logger.warning("No documents loaded.")
                return
            corpus = [doc["text"] for doc in self.documents]
            self.embeddings = self.vectorizer.fit_transform(corpus)
            self._initialized = True
            logger.info("[VectorStore] Indexing completed.")

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        self.build_index()
        if self.embeddings is None:
            return []
        cache_key = (query.lower(), top_k)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        query_vector = self.vectorizer.transform_query(query)
        similarity = self.vectorizer.cosine_similarity(query_vector, self.embeddings)
        ranked = np.argsort(similarity)[::-1]
        
        results = []
        for idx in ranked:
            score = float(similarity[idx])
            if score <= 0:
                continue
            document = dict(self.documents[idx])
            document["score"] = score
            results.append(document)
            if len(results) >= top_k:
                break
        self.cache[cache_key] = results
        return results

    def rerank(self, query: str, documents: List[dict], top_k: int = 3) -> List[dict]:
        if not documents:
            return []
        query_words = set(self.clean(query).split())
        ranked = []
        for doc in documents:
            doc_words = set(self.clean(doc["text"]).split())
            overlap = len(query_words & doc_words)
            final_score = doc["score"] + (overlap * 0.10)
            doc["rerank_score"] = final_score
            ranked.append(doc)
        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

    def add_documents(self, new_documents):
        if not new_documents:
            return
        self.documents.extend(new_documents)
        corpus = [d["text"] for d in self.documents]
        self.embeddings = self.vectorizer.fit_transform(corpus)
        self.cache.clear()
        logger.info("[VectorStore] Knowledge base index expanded.")

    def save_memory(self, session_id: str, text: str):
        with self._lock:
            if session_id not in self.memory:
                self.memory[session_id] = []
            self.memory[session_id].append(text)
            self.memory[session_id] = self.memory[session_id][-10:]

    def search_memory(self, query: str, session_id: str) -> str:
        if session_id not in self.memory:
            return ""
        query_words = set(self.clean(query).split())
        scored = []
        for memory in self.memory[session_id]:
            memory_words = set(self.clean(memory).split())
            overlap = len(query_words & memory_words)
            if overlap:
                scored.append((overlap, memory))
        scored.sort(reverse=True)
        return "\n".join(text for _, text in scored[:3])

    def clear_memory(self, session_id: str):
        with self._lock:
            if session_id in self.memory:
                del self.memory[session_id]

    def clear_cache(self):
        self.cache.clear()

    def reset(self):
        with self._lock:
            self.documents.clear()
            self.memory.clear()
            self.cache.clear()
            self.embeddings = None
            self._initialized = False
            logger.info("Vector store reset.")

# Singleton
_store = VectorStore()

# Public Functions
def init_vector_store():
    _store.build_index()

def retrieve(query: str, top_k: int = 5):
    return _store.retrieve(query=query, top_k=top_k)

def rerank(query: str, documents, top_k: int = 3):
    return _store.rerank(query=query, documents=documents, top_k=top_k)

def add_documents(documents):
    _store.add_documents(documents)

def save_memory(session_id: str, text: str):
    _store.save_memory(session_id, text)

def search_memory(query: str, session_id: str):
    return _store.search_memory(query, session_id)

def clear_memory(session_id: str):
    _store.clear_memory(session_id)

def clear_cache():
    _store.clear_cache()

def reset_vector_store():
    _store.reset()
