"""
backend/app/core/redis.py
----------------------------------------------------
Production-Grade Thread-Safe Redis Connection Manager & Cache for Genkit AI V6.1.
Supports distributed rate-limiting, response caching, and health diagnostics.
Gracefully falls back to zero-overhead in-memory operations if Redis is unavailable.
"""

import time
import json
import threading
from typing import Any, Dict, List, Optional, Tuple
from app.core.config import settings
from app.core.logger import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False


class RedisManager:
    """Thread-safe Redis Connection Manager with resilient in-memory fallback."""

    # Atomic Lua script for sliding-window rate limiting
    # ARGV: [now, window_start, limit, ttl]
    # Only adds the current timestamp if count is below the limit.
    SLIDING_WINDOW_LUA = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window_start = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local ttl = tonumber(ARGV[4])

    redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
    local current_count = redis.call('ZCARD', key)

    if current_count < limit then
        redis.call('ZADD', key, now, tostring(now))
        redis.call('EXPIRE', key, ttl)
        return 1
    else
        return 0
    end
    """

    def __init__(self):
        self._client: Optional[Any] = None
        self._is_connected: bool = False
        self._lock = threading.Lock()
        self._memory_cache: Dict[str, Tuple[float, Any]] = {}
        self._memory_rate_limits: Dict[str, List[float]] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Attempts connection to Redis if enabled and package is available."""
        if not getattr(settings, "REDIS_ENABLED", False):
            logger.info("RedisManager: Redis disabled via configuration (using in-memory fallback).")
            return

        if not REDIS_AVAILABLE or redis is None:
            logger.warning(
                "RedisManager: redis package not installed. Using in-memory fallback. "
                "Install with: pip install redis"
            )
            return

        with self._lock:
            try:
                host = getattr(settings, "REDIS_HOST", "localhost")
                port = int(getattr(settings, "REDIS_PORT", 6379))
                password = getattr(settings, "REDIS_PASSWORD", None) or None
                db = int(getattr(settings, "REDIS_DB", 0))

                pool = redis.ConnectionPool(
                    host=host,
                    port=port,
                    password=password,
                    db=db,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=True,
                )
                client = redis.Redis(connection_pool=pool)
                client.ping()
                self._client = client
                self._is_connected = True
                logger.info(f"RedisManager: Connected to Redis at {host}:{port}/{db}")
            except Exception as e:
                self._client = None
                self._is_connected = False
                logger.warning(f"RedisManager: Redis offline or unreachable ({e}). Using in-memory fallback.")

    @property
    def is_available(self) -> bool:
        return self._is_connected and self._client is not None

    @property
    def status_string(self) -> str:
        if not getattr(settings, "REDIS_ENABLED", False):
            return "disabled"
        return "active" if self.is_available else "unavailable"

    def ping(self) -> bool:
        """Pings Redis to check live connectivity."""
        if not self.is_available:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            self._is_connected = False
            return False

    def reconnect(self) -> bool:
        """Attempts to reconnect to Redis."""
        self._initialize()
        return self.is_available

    def check_rate_limit(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """
        Sliding-window rate limiter.
        Uses Redis atomic Lua script with sorted sets when available; falls back to in-memory sliding window.
        """
        now = time.time()
        window_start = now - window_seconds

        if self.is_available:
            try:
                redis_key = f"rate_limit:{key}"
                res = self._client.eval(
                    self.SLIDING_WINDOW_LUA,
                    1,
                    redis_key,
                    now,
                    window_start,
                    limit,
                    window_seconds + 5,
                )
                return bool(res == 1)
            except Exception as e:
                logger.warning(f"Redis rate limit check failed ({e}); falling back to memory.")

        # In-memory fallback
        with self._lock:
            timestamps = self._memory_rate_limits.setdefault(key, [])
            valid_timestamps = [t for t in timestamps if t > window_start]
            self._memory_rate_limits[key] = valid_timestamps
            if len(valid_timestamps) >= limit:
                return False
            valid_timestamps.append(now)
            return True

    def get_cache(self, key: str) -> Optional[Any]:
        """Retrieves cached object from Redis or in-memory dictionary."""
        if self.is_available:
            try:
                val = self._client.get(f"cache:{key}")
                if val is not None:
                    return json.loads(val)
            except Exception as e:
                logger.warning(f"Redis get_cache failed ({e}).")

        # In-memory fallback
        with self._lock:
            if key in self._memory_cache:
                expires_at, val = self._memory_cache[key]
                if time.time() < expires_at:
                    return val
                del self._memory_cache[key]
        return None

    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Stores object in Redis or in-memory dictionary with TTL in seconds."""
        if self.is_available:
            try:
                serialized = json.dumps(value)
                self._client.setex(f"cache:{key}", ttl, serialized)
                return True
            except Exception as e:
                logger.warning(f"Redis set_cache failed ({e}).")

        # In-memory fallback
        with self._lock:
            self._memory_cache[key] = (time.time() + ttl, value)
        return True

    def delete_cache(self, key: str) -> bool:
        """Deletes cached key from Redis or in-memory dictionary."""
        if self.is_available:
            try:
                self._client.delete(f"cache:{key}")
            except Exception as e:
                logger.warning(f"Redis delete_cache failed ({e}).")

        with self._lock:
            self._memory_cache.pop(key, None)
        return True


# Global singleton instance
redis_manager = RedisManager()
