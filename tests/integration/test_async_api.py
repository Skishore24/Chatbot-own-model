"""
tests/integration/test_async_api.py
----------------------------------------------------
Integration tests for FastAPI REST Endpoints & Real-time SSE Token Streaming.
"""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app


class TestAsyncAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "operational")

    def test_health_endpoint(self):
        response = self.client.get("/api/v5/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_chat_query_endpoint(self):
        payload = {"message": "What are Genkit AI services?", "session_id": "test_session_123"}
        response = self.client.post("/api/v5/chat/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertEqual(data["session_id"], "test_session_123")

    def test_sse_stream_endpoint(self):
        payload = {"message": "Tell me about AI development", "session_id": "test_sse_456"}
        response = self.client.post("/api/v5/chat/stream", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("data:", response.text)


if __name__ == "__main__":
    unittest.main()
