"""
backend/app/api/__init__.py
----------------------------------------------------
Master API router registration for Genkit AI V6.
"""

from fastapi import APIRouter

from app.api.chat import router as chat_router
from app.api.streaming import router as streaming_router
from app.api.leads import router as leads_router
from app.api.health import router as health_router
from app.api.history import router as history_router
from app.api.feedback import router as feedback_router
from app.api.auth import verify_api_key, verify_jwt_token

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(streaming_router)
api_v1_router.include_router(leads_router)
api_v1_router.include_router(history_router)
api_v1_router.include_router(feedback_router)

__all__ = ["api_v1_router", "verify_api_key", "verify_jwt_token"]

