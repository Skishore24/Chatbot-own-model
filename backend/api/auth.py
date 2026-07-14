"""
api/auth.py
----------------------------------------------------
Genkit AI - Authentication & Session Manager

Features
--------
• API Key Authentication
• Session ID Generation
• Session Validation
• UUID Verification
• Secure Random Token
• FastAPI Dependencies

Author : Genkit AI
"""

import os
import uuid
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("API_KEY", "").strip()

SESSION_LENGTH = 32


# ============================================================
# API KEY VERIFICATION
# ============================================================

def verify_api_key(
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key"
    )
):
    """
    Verify API Key.

    If API_KEY is empty,
    authentication is disabled.
    """

    if API_KEY == "":
        return True

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key"
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )

    return True


# ============================================================
# SESSION HELPERS
# ============================================================

def new_session_id() -> str:
    """
    Generate secure session id.
    """

    return str(uuid.uuid4())


def random_token(length: int = SESSION_LENGTH) -> str:
    """
    Generate random secure token.
    """

    return secrets.token_hex(length // 2)
# ============================================================
# SESSION VALIDATION
# ============================================================

def is_valid_session(session_id: Optional[str]) -> bool:
    """
    Check whether a session ID is a valid UUID.
    """

    if not session_id:
        return False

    try:
        uuid.UUID(str(session_id))
        return True
    except Exception:
        return False


def ensure_session(session_id: Optional[str]) -> str:
    """
    Return a valid session ID.
    Generate a new one if invalid.
    """

    if is_valid_session(session_id):
        return session_id.strip()

    return new_session_id()


def validate_uuid(session_id: str):
    """
    Raise HTTPException if UUID is invalid.
    """

    if not is_valid_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Session ID"
        )

    return session_id


# ============================================================
# REQUEST HELPERS
# ============================================================

def get_session(
    session_id: Optional[str] = Header(
        default=None,
        alias="X-Session-ID"
    )
):
    """
    FastAPI dependency.

    Reads session ID from request header.

    Header:
        X-Session-ID
    """

    return ensure_session(session_id)
# ============================================================
# EXTRA UTILITIES
# ============================================================

def generate_guest_id() -> str:
    """
    Generate guest identifier.

    Example:
        guest_4f8d3c8d7a91
    """

    return "guest_" + secrets.token_hex(6)


def session_info(session_id: Optional[str] = None) -> dict:
    """
    Return session information.
    """

    session_id = ensure_session(session_id)

    return {
        "session_id": session_id,
        "valid": is_valid_session(session_id),
        "type": "uuid"
    }


def auth_status() -> dict:
    """
    Returns authentication configuration.
    Useful for admin dashboard.
    """

    return {
        "authentication_enabled": bool(API_KEY),
        "session_header": "X-Session-ID",
        "api_key_header": "X-API-Key"
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "verify_api_key",
    "new_session_id",
    "ensure_session",
    "is_valid_session",
    "validate_uuid",
    "get_session",
    "random_token",
    "generate_guest_id",
    "session_info",
    "auth_status",
]
