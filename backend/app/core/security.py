"""
backend/app/core/security.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Zero-Trust Security Subsystem
Handles JWT Tokens, Input/Output Sanitization, SQLi & XSS Filtering, and Prompt Injection Scans.
"""

import re
import html
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from .config import settings
from .logger import logger


class SecurityService:
    """Enterprise Security & Sanitization Manager."""

    # Unsafe SQLi & XSS Patterns
    SQLI_PATTERNS = [
        re.compile(r"(\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|EXEC|EXECUTE)\b)", re.IGNORECASE),
        re.compile(r"(--|/\*|\*/|;\s*$)"),
        re.compile(r"(\bOR\b\s+1\s*=\s*1)", re.IGNORECASE),
    ]

    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
    ]

    # Prompt Injection Attack Patterns
    PROMPT_INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(a|an)\s+(unfiltered|dan|developer\s+mode)", re.IGNORECASE),
        re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.IGNORECASE),
        re.compile(r"disregard\s+safety\s+guidelines", re.IGNORECASE),
    ]

    def sanitize_input(self, text: str) -> Tuple[str, bool]:
        """
        Sanitizes input text for XSS and SQLi safety.
        Returns: (sanitized_text, is_safe)
        """
        if not text or not isinstance(text, str):
            return "", True

        # Strip HTML tags & encode entities
        cleaned = html.escape(text.strip())

        # Check SQLi
        for pattern in self.SQLI_PATTERNS:
            if pattern.search(cleaned):
                logger.warning(f"Security Alert: SQLi pattern detected in input: {text[:50]}...")
                return "", False

        # Check XSS
        for pattern in self.XSS_PATTERNS:
            if pattern.search(text):
                logger.warning(f"Security Alert: XSS pattern detected in input: {text[:50]}...")
                return "", False

        return cleaned, True

    def scan_prompt_injection(self, text: str) -> bool:
        """
        Scans input for prompt injection attack attempts.
        Returns: True if malicious injection is detected.
        """
        if not text:
            return False

        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning(f"Security Guard: Prompt injection attack flagged: {text[:50]}...")
                return True

        return False

    def sanitize_output(self, response_text: str) -> str:
        """
        Sanitizes LLM outputs to prevent tag leakage or unsafe code block emissions.
        """
        if not response_text:
            return ""

        # Remove leaked control tags
        cleaned = re.sub(r"<(context|question|answer|thought)[^>]*>.*?</\1>", "", response_text, flags=re.DOTALL)
        cleaned = re.sub(r"\[(ASSISTANT|USER|SYSTEM|CONTEXT)\]", "", cleaned)

        # Fix formatted email spacing issues
        cleaned = re.sub(r"(\w+)\s*@\s*(\w+)\s*\.\s*(\w+)", r"\1@\2.\3", cleaned)

        return cleaned.strip()

    def generate_token(self, payload: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Generates a lightweight HMAC-SHA256 signature token (Pure Python implementation without external JWT lib).
        """
        import base64
        import json

        header = {"alg": "HS256", "typ": "JWT"}
        exp = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        payload_copy = payload.copy()
        payload_copy["exp"] = int(exp.timestamp())

        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload_copy).encode()).decode().rstrip("=")

        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(settings.SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

        return f"{signing_input}.{signature_b64}"

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verifies and decodes a lightweight HMAC-SHA256 token.
        """
        import base64
        import json

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts
            signing_input = f"{header_b64}.{payload_b64}"

            # Re-compute signature
            expected_sig = hmac.new(settings.SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")

            if not hmac.compare_digest(signature_b64, expected_sig_b64):
                logger.warning("JWT verification failed: Invalid signature")
                return None

            # Decode payload
            padding = "=" * (4 - len(payload_b64) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64 + padding).decode()
            payload = json.loads(payload_json)

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.now(timezone.utc).timestamp() > exp:
                logger.warning("JWT verification failed: Token expired")
                return None

            return payload

        except Exception as e:
            logger.error(f"Error decoding security token: {str(e)}")
            return None


    def __init__(self):
        self._rate_limits: Dict[str, List[float]] = {}

    def check_rate_limit(self, client_ip: str, limit_per_minute: Optional[int] = None) -> bool:
        """
        Sliding-window per-IP rate limiter.
        Returns True if request is allowed, False if rate limit exceeded.
        """
        import time
        limit = limit_per_minute or settings.RATE_LIMIT_PER_MINUTE
        now = time.time()
        window_start = now - 60.0

        if client_ip not in self._rate_limits:
            self._rate_limits[client_ip] = []

        # Retain only timestamps within the last 60 seconds
        self._rate_limits[client_ip] = [t for t in self._rate_limits[client_ip] if t > window_start]

        if len(self._rate_limits[client_ip]) >= limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip} ({len(self._rate_limits[client_ip])}/{limit} req/min)")
            return False

        self._rate_limits[client_ip].append(now)
        return True


security_service = SecurityService()
