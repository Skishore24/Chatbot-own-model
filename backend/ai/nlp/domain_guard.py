"""
ai/nlp/domain_guard.py
----------------------------------------------------
Genkit AI - Domain Guard

Purpose
-------
Determines whether a user's query is within the Genkit domain
before passing it to the LLM. Acts as a fast, cheap first gate.

Logic
-----
1.  If the query contains a strong Genkit-positive keyword → in-domain.
2.  If the query contains an out-of-scope keyword → out-of-domain.
3.  Otherwise → uncertain (treated as in-domain for safety).

The guard does NOT block uncertain queries — the LLM itself
handles graceful refusals if context cannot be retrieved.

Author : Genkit AI
"""

import re
import logging
from typing import Dict, Set

logger = logging.getLogger("Genkit AI")

# ============================================================
# GENKIT DOMAIN KEYWORDS
# ============================================================
# Words that strongly indicate the user is asking about Genkit
_GENKIT_POSITIVE: Set[str] = {
    "genkit",
    "website", "web", "webpage", "webpage", "webdev",
    "chatbot", "ai", "bot", "automation",
    "design", "logo", "branding", "brand",
    "seo", "search engine", "ranking",
    "marketing", "campaign",
    "video", "editing", "reel", "short",
    "mobile", "android", "ios", "app",
    "software", "application", "platform",
    "portfolio", "project", "projects",
    "hosting", "domain", "deploy",
    "figma", "photoshop", "illustrator",
    "premiere", "after effects",
    "python", "nodejs", "node", "java", "react",
    "mysql", "mongodb", "database",
    "price", "pricing", "cost", "quote", "quotation",
    "service", "services", "offer",
    "contact", "email", "phone", "reach",
    "team", "founder", "company", "startup",
    "mission", "vision", "tagline", "motto",
    "hire", "work", "project",
    "instagram", "youtube", "twitter", "github",
    "support", "maintenance", "update",
    "ecommerce", "e-commerce", "shop", "store",
    "landing page", "banner", "flyer", "thumbnail",
    "motion graphics", "animation",
    "responsive", "mobile friendly",
    "ui", "ux", "interface", "prototype", "mockup",
    "backend", "frontend", "fullstack", "full stack",
    "api", "rest", "integration",
    "ssl", "security", "https",
    "google", "rank",
    "client", "customer", "business",
    "affordable", "budget",
    "nda", "consultation", "free consultation",
    "international", "global",
    "wordpress", "cms",
    "subscription", "package", "deal",
    "refund", "revision",
    "turnaround", "delivery", "deadline",
    "genkit.in", "genkit.tech",
    "rag", "llm", "gpt", "model", "custom model", "tokenizer", "embedding", "transformer", "inference", "architecture",
}

# ============================================================
# OUT-OF-SCOPE KEYWORDS
# ============================================================
# Words that indicate the query has nothing to do with Genkit
_OUT_OF_SCOPE: Set[str] = {
    "weather", "temperature", "forecast", "rain", "snow",
    "cricket", "football", "soccer", "baseball", "basketball",
    "ipl", "fifa", "nfl", "nba", "sports", "match", "score",
    "movie", "film", "actor", "actress", "cinema", "netflix",
    "song", "music", "singer", "singer", "album", "band",
    "bitcoin", "crypto", "ethereum", "nft", "blockchain",
    "stock market", "shares", "investment", "forex", "trading",
    "politics", "election", "vote", "politician",
    "president", "prime minister", "minister", "government",
    "recipe", "cook", "food", "restaurant", "diet",
    "medicine", "doctor", "hospital", "drug", "disease",
    "covid", "pandemic", "vaccine",
    "history", "geography", "science", "physics", "chemistry",
    "mathematics", "algebra", "calculus",
    "book", "novel", "author", "literature",
    "yoga", "gym", "fitness", "exercise", "workout",
    "fashion", "clothes", "style",
    "travel", "tourism", "vacation", "hotel", "flight",
    "joke", "funny", "humor", "meme",
    "poem", "poetry", "story", "essay",
    "homework", "exam", "test", "study",
    "translate", "translation", "language",
    "religion", "god", "prayer",
    "animal", "pet", "cat", "dog",
    "celebrity", "famous person", "star",
    "news", "headline", "current events",
    "lottery", "casino", "gambling",
    "horoscope", "zodiac", "astrology",
    "space", "nasa", "planet",
}

# ============================================================
# GENKIT ENTITY PATTERNS
# ============================================================
# Patterns that make a query very likely to be Genkit-related
_GENKIT_PATTERNS = [
    re.compile(r"\bgenkit\b", re.IGNORECASE),
    re.compile(r"\bgenkit\.in\b", re.IGNORECASE),
    re.compile(r"\bgenkit\.tech\b", re.IGNORECASE),
    re.compile(r"\byour\s+(service|website|company|team|price|pricing)\b", re.IGNORECASE),
    re.compile(r"\byou\s+(offer|provide|do|build|make|create|develop)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+you\b", re.IGNORECASE),
    re.compile(r"\bcan\s+you\b", re.IGNORECASE),
    re.compile(r"\bhow\s+much\s+(do\s+you|does|will)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+your\b", re.IGNORECASE),
    re.compile(r"\bwho\s+(are|founded|is\s+genkit)\b", re.IGNORECASE),
    re.compile(r"\bcontact\s+(you|genkit)\b", re.IGNORECASE),
    re.compile(r"\bhire\s+(you|genkit)\b", re.IGNORECASE),
]


class DomainGuard:
    """
    Fast domain classifier for Genkit AI.

    Determines whether a user query is Genkit-related before
    passing it to the expensive LLM inference pipeline.

    Returns
    -------
    dict with keys:
        in_domain   : bool
        confidence  : float  (0.0 - 1.0)
        reason      : str
    """

    def __init__(self) -> None:
        self.positive_keywords = _GENKIT_POSITIVE
        self.oos_keywords = _OUT_OF_SCOPE
        self.patterns = _GENKIT_PATTERNS

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _tokenize(self, text: str) -> Set[str]:
        """Lower-case word tokenisation."""
        return set(re.findall(r"[a-z]+", text.lower()))

    def _has_pattern_match(self, text: str) -> bool:
        for pattern in self.patterns:
            if pattern.search(text):
                return True
        return False

    # ----------------------------------------------------------
    # Public classify
    # ----------------------------------------------------------

    def classify(self, query: str) -> Dict:
        """
        Classify whether query is in Genkit's domain.

        Parameters
        ----------
        query : str  Raw user input.

        Returns
        -------
        dict
            {
                "in_domain" : bool,
                "confidence": float,
                "reason"    : str
            }
        """
        if not query or not query.strip():
            return {
                "in_domain": False,
                "confidence": 1.0,
                "reason": "empty_query",
            }

        tokens = self._tokenize(query)

        # 1. Pattern match (highest confidence)
        if self._has_pattern_match(query):
            return {
                "in_domain": True,
                "confidence": 0.95,
                "reason": "pattern_match",
            }

        # 2. Positive keyword match
        positive_hits = tokens & self.positive_keywords
        if positive_hits:
            confidence = min(0.90, 0.50 + len(positive_hits) * 0.15)
            return {
                "in_domain": True,
                "confidence": round(confidence, 2),
                "reason": f"keyword_match:{','.join(list(positive_hits)[:3])}",
            }

        # 3. Out-of-scope keyword match
        oos_hits = tokens & self.oos_keywords
        if oos_hits:
            confidence = min(0.95, 0.60 + len(oos_hits) * 0.15)
            return {
                "in_domain": False,
                "confidence": round(confidence, 2),
                "reason": f"oos_keyword:{','.join(list(oos_hits)[:3])}",
            }

        # 4. No signal → treat as uncertain / in-domain
        #    The LLM will respond gracefully if it finds no context.
        return {
            "in_domain": True,
            "confidence": 0.40,
            "reason": "uncertain",
        }

    def is_in_domain(self, query: str) -> bool:
        """Convenience method: returns True if query is in-domain."""
        return self.classify(query)["in_domain"]

    def get_refusal_message(self) -> str:
        """Standard polite refusal for out-of-scope queries."""
        return (
            "I am Genkit AI, designed exclusively to assist with questions "
            "about Genkit's digital services, company, and team. "
            "I am not able to help with this topic. "
            "Please feel free to ask me anything about Genkit!"
        )


# ============================================================
# Singleton
# ============================================================
domain_guard = DomainGuard()

__all__ = ["DomainGuard", "domain_guard"]
