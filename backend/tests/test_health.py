"""
backend/tests/test_health.py
----------------------------------------------------
Tests for /api/v1/health and /api/v1/model diagnostic endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint_schema(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    assert "model" in data
    assert "status" in data["model"]
    assert "rag" in data
    assert "database" in data
    assert data["rag"]["documents"] > 0


def test_model_info_endpoint(client):
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "Genkit Enterprise GPT v6.1"
    assert data["vocab_size"] == 2084
    assert data["embed_dim"] == 384
    assert data["num_layers"] == 6
    assert data["num_heads"] == 6


def test_rag_knowledge_endpoint(client):
    response = client.get("/api/v1/rag/knowledge")
    assert response.status_code == 200
    chunks = response.json()
    assert isinstance(chunks, list)
    assert len(chunks) >= 40
