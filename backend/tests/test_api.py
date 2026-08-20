"""
backend/tests/test_api.py
----------------------------------------------------
Integration tests for FastAPI endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_and_health(self):
        """Test root and health check endpoints."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "operational")

        health_res = self.client.get("/api/v1/health")
        self.assertEqual(health_res.status_code, 200)
        h_data = health_res.json()
        self.assertEqual(h_data["status"], "healthy")
        self.assertIn("model", h_data)
        self.assertIn("rag", h_data)
        self.assertIn("database", h_data)

    def test_model_info(self):
        """Test model info endpoint."""
        res = self.client.get("/api/v1/model")
        self.assertEqual(res.status_code, 200)
        self.assertIn("vocab_size", res.json())

    def test_chat_in_domain(self):
        """Test in-domain chat endpoint."""
        payload = {"message": "What services does Genkit provide?", "session_id": "test_session_1"}
        res = self.client.post("/api/v1/chat", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["grounded"])
        self.assertTrue(len(data["answer"]) > 5)

    def test_chat_stream_endpoint(self):
        """Test SSE streaming endpoint."""
        payload = {"message": "What services does Genkit provide?", "session_id": "test_stream_1"}
        res = self.client.post("/api/v1/chat/stream", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers.get("content-type", ""))
        body = res.text
        self.assertIn("data: ", body)
        self.assertIn('"event": "start"', body)
        self.assertIn('"event": "end"', body)

    def test_chat_out_of_domain(self):
        """Test out-of-domain chat query refusal."""
        payload = {"message": "What is the capital of France?", "session_id": "test_session_2"}
        res = self.client.post("/api/v1/chat", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["grounded"])
        self.assertIn("verified information", data["answer"])

    def test_lead_capture(self):
        """Test lead capture endpoint."""
        payload = {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "+1234567890",
            "message": "Interested in custom AI development.",
            "session_id": "test_session_1",
        }
        res = self.client.post("/api/v1/leads", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])

    def test_chat_history(self):
        """Test history retrieval for session."""
        res = self.client.get("/api/v1/history?session_id=test_session_1")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["session_id"], "test_session_1")
        self.assertTrue(isinstance(data["messages"], list))


if __name__ == "__main__":
    unittest.main()
