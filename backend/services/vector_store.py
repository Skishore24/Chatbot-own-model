import chromadb
from chromadb.utils import embedding_functions
import os
import hashlib
from typing import List, Optional
from app.config import CHROMA_PATH, logger

# ─────────────────────────────────────────────
# CHROMA & EMBEDDINGS SETUP
# ─────────────────────────────────────────────
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

try:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # 1. Knowledge Collection (Static company data)
    try:
        kb_collection = client.get_or_create_collection(
            name="genkit_docs",
            embedding_function=embedding_function
        )
    except ValueError:
        # Handle embedding function conflict if it arises
        client.delete_collection(name="genkit_docs")
        kb_collection = client.create_collection(
            name="genkit_docs",
            embedding_function=embedding_function
        )

    # 2. Semantic Memory Collection (Dynamic session history)
    try:
        memory_collection = client.get_or_create_collection(
            name="semantic_memory",
            embedding_function=embedding_function
        )
    except ValueError:
        client.delete_collection(name="semantic_memory")
        memory_collection = client.create_collection(
            name="semantic_memory",
            embedding_function=embedding_function
        )

except Exception as e:
    logger.error(f"Critical Error: Failed to initialize ChromaDB: {e}")
    raise

# ─────────────────────────────────────────────
# KNOWLEDGE BASE LOGIC
# ─────────────────────────────────────────────
def load_and_split() -> List[str]:
    """Reads company.txt and splits into logical chunks."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    file_path = os.path.join(data_dir, "company.txt")
    
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split by logical markers or fallback to paragraphs
        chunks = [s.strip() for s in text.split("--------------------------------------------------") if s.strip()]
        if not chunks:
            chunks = [s.strip() for s in text.split("\n\n") if s.strip()]
        return chunks
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
        return []

def add_documents(docs: List[str]):
    """Indexes documents in the knowledge collection."""
    if not docs: return
    try:
        ids = [hashlib.md5(doc.encode()).hexdigest() for doc in docs]
        kb_collection.add(documents=docs, ids=ids)
        logger.info(f"Vector Store: Indexed {len(docs)} knowledge chunks.")
    except Exception as e:
        logger.error(f"Indexing error: {e}")

def keyword_match_score(query: str, doc: str) -> float:
    """Calculates a simple overlap score for reinforcement."""
    q_words = set(query.lower().split())
    d_words = set(doc.lower().split())
    return len(q_words & d_words)

def search(query: str, n_results: int = 5, threshold: float = 0.8) -> str:
    """Search knowledge base with hybrid scoring (Vector + Keyword)."""
    try:
        results = kb_collection.query(query_texts=[query], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        scored = []
        for doc, dist in zip(docs, distances):
            if dist < threshold:
                # Higher keyword match boosts the score
                final_score = (1 - dist) + (0.2 * keyword_match_score(query, doc))
                scored.append((doc, final_score))
        
        if not scored: return ""
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return "\n\n".join([d[0] for d in scored[:2]]).strip()
    except Exception as e:
        logger.error(f"KB Search error: {e}")
        return ""

# ─────────────────────────────────────────────
# SEMANTIC MEMORY LOGIC
# ─────────────────────────────────────────────
def save_memory(session_id: str, text: str):
    """Saves an interaction to the semantic memory for future context."""
    try:
        # Create a unique but session-traceable ID
        mem_id = f"{session_id}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        memory_collection.add(
            documents=[text],
            metadatas=[{"session_id": session_id}],
            ids=[mem_id]
        )
    except Exception as e:
        logger.debug(f"Failed to save semantic memory: {e}")

def search_memory(query: str, session_id: str, n_results: int = 2) -> str:
    """Retrieves relevant past interactions for the current session."""
    try:
        results = memory_collection.query(
            query_texts=[query],
            where={"session_id": session_id},
            n_results=n_results
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs).strip()
    except Exception as e:
        logger.error(f"Memory Search error: {e}")
        return ""