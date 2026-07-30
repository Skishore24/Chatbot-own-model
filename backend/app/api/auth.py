"""
backend/app/api/auth.py
----------------------------------------------------
GENKIT AI v5.0 API Authentication & Authorization Middleware
"""

from fastapi import Depends, HTTPException, Header, status
from typing import Optional

from app.core.config import settings
from app.core.security import security_service


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Verifies optional API Key header in production requests."""
    if not settings.API_KEY:
        return True  # Open key in development mode

    if x_api_key and x_api_key == settings.API_KEY:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing X-API-Key header.",
    )


async def verify_jwt_token(authorization: Optional[str] = Header(None)) -> dict:
    """Verifies JWT Bearer token header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization Bearer header.",
        )

    token = authorization.split(" ")[1]
    payload = security_service.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
        )

    return payload
