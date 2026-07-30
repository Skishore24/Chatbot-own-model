"""
backend/tests/test_async_api.py
----------------------------------------------------
Unit tests for FastAPI REST Endpoints, SSE Token Streaming, and Health Checks.
"""

import sys
import unittest
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app


class TestAsyncAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Verify root endpoint status."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "operational")

    def test_health_endpoint(self):
        """Verify health check endpoint."""
        response = self.client.get("/api/v5/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_chat_query_endpoint(self):
        """Verify synchronous chat query REST endpoint."""
        payload = {"message": "What are Genkit AI services?", "session_id": "test_session_123"}
        response = self.client.post("/api/v5/chat/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertEqual(data["session_id"], "test_session_123")

    def test_sse_stream_endpoint(self):
        """Verify SSE token streaming HTTP endpoint."""
        payload = {"message": "Tell me about AI development", "session_id": "test_sse_456"}
        response = self.client.post("/api/v5/chat/stream", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("data:", response.text)


if __name__ == "__main__":
    unittest.main()
