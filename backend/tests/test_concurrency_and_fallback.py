"""
backend/tests/test_concurrency_and_fallback.py
----------------------------------------------------
Tests for graceful degradation: missing model checkpoint handling, fallback to verified RAG, and concurrent request handling.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from app.main import app
from app.llm.inference import load_model_and_tokenizer, ModelStatus


class TestConcurrencyAndFallback(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_missing_model_graceful_handling(self):
        """Test that missing checkpoint returns ModelStatus.NOT_TRAINED without crashing."""
        model, tokenizer, config, status = load_model_and_tokenizer(model_path="non_existent_model.pt")
        self.assertIsNone(model)
        self.assertIn(status, [ModelStatus.NOT_FOUND, ModelStatus.NOT_TRAINED])
        self.assertIsNotNone(tokenizer)

    def test_concurrent_chat_requests(self):
        """Test handling multiple simultaneous chat requests."""
        def make_request(idx):
            payload = {
                "message": f"What is Genkit? ({idx})",
                "session_id": f"concurrent_test_{idx}",
            }
            return self.client.post("/api/v1/chat", json=payload)

        with ThreadPoolExecutor(max_workers=5) as executor:
            responses = list(executor.map(make_request, range(5)))

        for resp in responses:
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertTrue(data["grounded"])
            self.assertTrue(len(data["answer"]) > 0)


if __name__ == "__main__":
    unittest.main()
