"""
backend/tests/test_langchain.py
----------------------------------------------------
Tests for LangChain LLM Adapter, BaseRetriever, LCEL Chains, and API Endpoints.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from langchain_core.documents import Document
from langchain_core.outputs import GenerationChunk

from app.main import app
from app.llm.langchain_wrapper import GenkitLLM
from app.rag.langchain_rag import GenkitRAGRetriever, format_docs, create_genkit_rag_chain


@pytest.fixture
def client():
    return TestClient(app)


def test_genkit_llm_properties():
    """Verify GenkitLLM identifies itself correctly to LangChain."""
    llm = GenkitLLM(temperature=0.5, top_k=40, max_new_tokens=64)
    assert llm._llm_type == "genkit_custom_llm"
    params = llm._identifying_params
    assert params["temperature"] == 0.5
    assert params["top_k"] == 40
    assert params["max_new_tokens"] == 64


def test_genkit_llm_call():
    """Verify GenkitLLM generates text or falls back cleanly."""
    llm = GenkitLLM(max_new_tokens=32)
    response = llm.invoke("What is Genkit?")
    assert isinstance(response, str)
    assert len(response) > 5


def test_genkit_llm_stream():
    """Verify GenkitLLM streaming yields GenerationChunks."""
    llm = GenkitLLM(max_new_tokens=32)
    chunks = list(llm.stream("Hello"))
    assert len(chunks) > 0
    assert all(isinstance(c, str) or isinstance(c, GenerationChunk) for c in chunks)


def test_genkit_rag_retriever():
    """Verify GenkitRAGRetriever returns standard LangChain Document objects."""
    retriever = GenkitRAGRetriever(top_k=2)
    docs = retriever.invoke("web development technologies")
    assert isinstance(docs, list)
    assert len(docs) > 0
    for doc in docs:
        assert isinstance(doc, Document)
        assert len(doc.page_content) > 10
        assert "id" in doc.metadata
        assert "title" in doc.metadata


def test_format_docs():
    """Verify format_docs converts documents into numbered context strings."""
    docs = [
        Document(page_content="Content 1", metadata={"title": "Doc 1"}),
        Document(page_content="Content 2", metadata={"title": "Doc 2"}),
    ]
    formatted = format_docs(docs)
    assert "[1] Doc 1\nContent 1" in formatted
    assert "[2] Doc 2\nContent 2" in formatted


def test_lcel_rag_chain_execution():
    """Verify full LangChain Expression Language (LCEL) chain executes correctly."""
    chain = create_genkit_rag_chain()
    result = chain.invoke("What services does Genkit offer?")
    assert isinstance(result, str)
    assert len(result.strip()) > 10


def test_langchain_api_info_endpoint(client):
    """Verify /api/v1/langchain/info returns operational metadata."""
    response = client.get("/api/v1/langchain/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["framework"] == "LangChain"
    assert "langchain_version" in data
    assert "LCEL Chains (Runnable)" in data["supported_features"]


def test_langchain_api_chat_endpoint(client):
    """Verify /api/v1/langchain/chat executes an LCEL chain over HTTP."""
    response = client.post(
        "/api/v1/langchain/chat",
        json={"message": "What is Genkit AI?", "temperature": 0.7},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Genkit" in data["answer"] or len(data["answer"]) > 10
    assert data["framework"] == "LangChain LCEL"
    assert len(data["sources"]) > 0

