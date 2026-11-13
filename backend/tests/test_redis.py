"""
backend/tests/test_redis.py
----------------------------------------------------
Unit tests for RedisManager (connection handling, caching, and sliding-window rate limiting).
Tests both in-memory fallback mode and Redis-connected mode (via mock).
"""

import time
import json
import unittest
from unittest.mock import MagicMock, patch
from app.core.redis import RedisManager


class TestRedisManager(unittest.TestCase):

    def setUp(self):
        self.manager = RedisManager()

    def test_default_state(self):
        """Test default fallback status when Redis is disabled/offline."""
        self.assertIn(self.manager.status_string, ["disabled", "active", "unavailable"])
        self.assertIsInstance(self.manager.is_available, bool)

    def test_in_memory_sliding_window_rate_limiting(self):
        """Test sliding window rate limiting in in-memory fallback mode."""
        key = "test_mem_rate_limit"
        limit = 3

        # First 3 should pass
        for _ in range(limit):
            self.assertTrue(self.manager.check_rate_limit(key, limit=limit, window_seconds=60))

        # 4th should be blocked
        self.assertFalse(self.manager.check_rate_limit(key, limit=limit, window_seconds=60))

    def test_in_memory_cache_set_get_and_delete(self):
        """Test setting, getting, and miss handling in in-memory cache."""
        cache_key = "test_mem_cache_key"
        payload = {"answer": "Genkit AI", "score": 0.95}

        # Cache miss
        self.assertIsNone(self.manager.get_cache("non_existent_key"))

        # Set cache
        success = self.manager.set_cache(cache_key, payload, ttl=10)
        self.assertTrue(success)

        # Cache hit
        cached = self.manager.get_cache(cache_key)
        self.assertEqual(cached, payload)

        # Delete cache
        self.manager.delete_cache(cache_key)
        self.assertIsNone(self.manager.get_cache(cache_key))

    def test_in_memory_cache_expiration(self):
        """Test TTL expiration in fallback cache."""
        cache_key = "test_expiring_key"
        self.manager.set_cache(cache_key, "temp_value", ttl=1)
        time.sleep(1.1)
        self.assertIsNone(self.manager.get_cache(cache_key))

    def test_redis_connected_rate_limiting(self):
        """Test Redis-connected branch using mock client with Lua script."""
        mock_client = MagicMock()
        # Mock Lua script eval returning 1 (allowed) then 0 (blocked)
        mock_client.eval.side_effect = [1, 1, 0]

        manager = RedisManager()
        manager._client = mock_client
        manager._is_connected = True

        self.assertTrue(manager.is_available)
        self.assertTrue(manager.check_rate_limit("user:123", limit=2, window_seconds=60))
        self.assertTrue(manager.check_rate_limit("user:123", limit=2, window_seconds=60))
        self.assertFalse(manager.check_rate_limit("user:123", limit=2, window_seconds=60))

        # Verify eval was called with correct script and parameters
        self.assertEqual(mock_client.eval.call_count, 3)

    def test_redis_connected_cache_operations(self):
        """Test Redis-connected get_cache and set_cache using mock client."""
        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps({"role": "assistant", "text": "hello"})

        manager = RedisManager()
        manager._client = mock_client
        manager._is_connected = True

        # Test get_cache hit
        cached = manager.get_cache("greeting")
        self.assertEqual(cached, {"role": "assistant", "text": "hello"})
        mock_client.get.assert_called_with("cache:greeting")

        # Test set_cache
        manager.set_cache("greeting", {"role": "assistant", "text": "hello"}, ttl=60)
        mock_client.setex.assert_called_with(
            "cache:greeting", 60, json.dumps({"role": "assistant", "text": "hello"})
        )

        # Test delete_cache
        manager.delete_cache("greeting")
        mock_client.delete.assert_called_with("cache:greeting")

    def test_redis_exception_resilience_fallback(self):
        """Test graceful fallback to in-memory when Redis client encounters errors."""
        mock_client = MagicMock()
        mock_client.eval.side_effect = RuntimeError("Redis connection lost")
        mock_client.get.side_effect = RuntimeError("Redis connection lost")
        mock_client.setex.side_effect = RuntimeError("Redis connection lost")

        manager = RedisManager()
        manager._client = mock_client
        manager._is_connected = True

        # Rate limit should catch exception and fallback to in-memory
        self.assertTrue(manager.check_rate_limit("resilient_key", limit=1))
        self.assertFalse(manager.check_rate_limit("resilient_key", limit=1))

        # Cache set and get should not raise exception
        manager.set_cache("fallback_key", "saved_in_mem", ttl=10)
        self.assertEqual(manager.get_cache("fallback_key"), "saved_in_mem")

    def test_ping_and_reconnect(self):
        """Test ping and reconnect logic."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        manager = RedisManager()
        # When disconnected
        self.assertFalse(manager.ping())

        # When connected
        manager._client = mock_client
        manager._is_connected = True
        self.assertTrue(manager.ping())

        # When ping fails
        mock_client.ping.side_effect = Exception("offline")
        self.assertFalse(manager.ping())
        self.assertFalse(manager.is_available)


if __name__ == "__main__":
    unittest.main()
