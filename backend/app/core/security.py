"""
backend/app/core/security.py
----------------------------------------------------
GENKIT AI v6.0 Security Subsystem
Handles input validation, prompt injection scanning, sliding-window rate limiting,
and JWT signature tokens without external dependencies.
"""

import re
import time
import base64
import json
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .logger import logger


class SecurityService:
    """Security & Validation Manager for Genkit AI V6."""

    # Malicious injection and exploit patterns
    PROMPT_INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(a|an)\s+(unfiltered|dan|developer\s+mode)", re.IGNORECASE),
        re.compile(r"reveal\s+(your\s+)?system\s+(prompt|instructions)", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(safety|system)\s+guidelines", re.IGNORECASE),
        re.compile(r"print\s+(your\s+)?initial\s+prompt", re.IGNORECASE),
    ]

    # Explicit SQL injection patterns for query validation
    SQLI_PATTERNS = [
        re.compile(r"(\bUNION\s+SELECT\b)", re.IGNORECASE),
        re.compile(r"(;\s*DROP\s+TABLE\b)", re.IGNORECASE),
        re.compile(r"(\bOR\b\s+1\s*=\s*1\s*--)", re.IGNORECASE),
    ]

    def __init__(self):
        self._rate_limits: Dict[str, List[float]] = {}

    def sanitize_input(self, text: str) -> Tuple[str, bool]:
        """
        Validates and cleans input text.
        NOTE: Does NOT HTML-escape text, preserving raw user characters for BM25/TF-IDF retrieval.
        Returns: (clean_text, is_safe)
        """
        if not text or not isinstance(text, str):
            return "", True

        # Remove null bytes and invisible control characters
        cleaned = text.replace("\x00", "").strip()

        # Check maximum length
        if len(cleaned) > settings.MAX_PROMPT_LENGTH:
            cleaned = cleaned[: settings.MAX_PROMPT_LENGTH]

        # Scan for explicit SQL injection strings in freeform text
        for pattern in self.SQLI_PATTERNS:
            if pattern.search(cleaned):
                logger.warning(f"Security Alert: SQLi pattern detected in query: {cleaned[:40]}...")
                return "", False

        return cleaned, True

    def scan_prompt_injection(self, text: str) -> bool:
        """
        Scans input for adversarial prompt injection attempts.
        Returns: True if malicious injection is detected.
        """
        if not text:
            return False

        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning(f"Security Guard: Prompt injection attempt blocked: {text[:60]}...")
                return True

        return False

    def sanitize_output(self, response_text: str) -> str:
        """
        Sanitizes model output to prevent special control tag leakage.
        """
        if not response_text:
            return ""

        # Strip internal special tokens and tag leaks
        cleaned = re.sub(r"<[a-zA-Z0-9_]+>", "", response_text)
        cleaned = re.sub(r"</[a-zA-Z0-9_]+>", "", cleaned)
        cleaned = re.sub(r"\[(ASSISTANT|USER|SYSTEM|CONTEXT)\]", "", cleaned)
        return cleaned.strip()

    def check_rate_limit(self, client_ip: str, limit_per_minute: Optional[int] = None) -> bool:
        """
        Sliding-window per-IP rate limiter.
        Returns True if request is allowed, False if rate limit exceeded.
        """
        limit = limit_per_minute or settings.RATE_LIMIT_PER_MINUTE
        now = time.time()
        window_start = now - 60.0

        if client_ip not in self._rate_limits:
            self._rate_limits[client_ip] = []

        # Keep timestamps within the last 60 seconds
        self._rate_limits[client_ip] = [t for t in self._rate_limits[client_ip] if t > window_start]

        if len(self._rate_limits[client_ip]) >= limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip} ({len(self._rate_limits[client_ip])}/{limit} req/min)")
            return False

        self._rate_limits[client_ip].append(now)
        return True

    def generate_token(self, payload: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Lightweight pure-Python HMAC-SHA256 token generator."""
        header = {"alg": "HS256", "typ": "JWT"}
        exp = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=24))
        payload_copy = payload.copy()
        payload_copy["exp"] = int(exp.timestamp())

        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload_copy).encode()).decode().rstrip("=")

        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(settings.SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

        return f"{signing_input}.{signature_b64}"

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Lightweight pure-Python HMAC-SHA256 token verifier."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts
            signing_input = f"{header_b64}.{payload_b64}"

            expected_sig = hmac.new(settings.SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")

            if not hmac.compare_digest(signature_b64, expected_sig_b64):
                return None

            padding = "=" * (4 - len(payload_b64) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64 + padding).decode()
            payload = json.loads(payload_json)

            exp = payload.get("exp")
            if exp and datetime.now(timezone.utc).timestamp() > exp:
                return None

            return payload
        except Exception:
            return None


security_service = SecurityService()
