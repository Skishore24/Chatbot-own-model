"""
backend/app/core/__init__.py
----------------------------------------------------
Core configuration, logging, and security exports.
"""

from app.core.config import settings
from app.core.logger import logger, get_logger, new_trace_id
from app.core.security import security_service
from app.core.redis import redis_manager

__all__ = [
    "settings",
    "logger",
    "get_logger",
    "new_trace_id",
    "security_service",
    "redis_manager",
]
