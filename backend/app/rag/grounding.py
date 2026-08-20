"""
backend/app/rag/grounding.py
----------------------------------------------------
Grounding Validator & Domain Refusal Guard for Genkit AI V6.1.
- Keyword domain guard + RAG confidence fusion
- Answer-level grounding validation against retrieved context chunks
"""

import re
from typing import List, Tuple
from app.rag.chunker import DocumentChunk
from app.rag.index import normalize_tokens

DOMAIN_REFUSAL_MESSAGE = (
    "I can help with Genkit's company, services, projects, technologies, pricing, "
    "and contact information. I don't have verified information about that topic."
)

GENKIT_CORE_KEYWORDS = {
    # Company & team
    "genkit", "company", "agency", "founder", "founders", "team",
    "location", "mission", "vision", "values", "policy", "faq", "india", "remote",
    "kishore", "hari", "dharani", "deepak", "rahul", "jithesh", "dharanesh", "ananya",
    # Services
    "service", "services", "web", "website", "mobile", "app", "apps", "ai", "ml",
    "chatbot", "automation", "ui", "ux", "design", "graphic", "branding", "logo",
    "seo", "marketing", "digital", "video", "editing", "motion", "animation",
    "ecommerce", "store", "saas", "portal", "custom", "software", "development",
    # Technologies
    "technology", "technologies", "stack", "react", "python", "pytorch", "fastapi",
    "django", "node", "nodejs", "mysql", "mongodb", "postgresql", "figma",
    "photoshop", "illustrator", "premiere", "effects", "aws", "cloud",
    # Pricing & Process
    "price", "pricing", "cost", "quote", "rate", "rates", "package", "packages",
    "budget", "hourly", "timeline", "turnaround", "process", "consultation",
    "hire", "contact", "email", "phone", "support", "revisions", "nda", "refund",
    # Conversational intent
    "build", "create", "offer", "provide", "portfolio", "projects", "case",
}


class GroundingValidator:
    """Validates domain relevance, computes groundedness, and validates answer facts."""

    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold

    def is_in_domain(self, query: str, top_retrieval_score: float = 0.0) -> bool:
        """
        Determines if the user query is relevant using keyword presence + retrieval confidence.
        """
        query_tokens = normalize_tokens(query)
        if not query_tokens:
            return True

        # Conversational greetings / short query check
        if len(query_tokens) <= 3 and any(t in {"hi", "hello", "hey", "help", "who", "what"} for t in query_tokens):
            return True

        # Check keyword matches
        matches = [t for t in query_tokens if t in GENKIT_CORE_KEYWORDS]
        if len(matches) > 0:
            return True

        # High RAG retrieval confidence indicates domain match even with new vocabulary
        if top_retrieval_score >= 2.5:
            return True

        return False

    def compute_grounding_score(
        self, query: str, chunks: List[DocumentChunk], top_retrieval_score: float = 0.0
    ) -> Tuple[float, bool]:
        """
        Computes grounding confidence score and domain validity.
        Returns: (confidence_score [0.0..1.0], is_grounded)
        """
        if not chunks:
            return 0.0, False

        query_tokens = set(normalize_tokens(query))
        if not query_tokens:
            return 1.0, True

        # Combine text of retrieved top chunks
        context_tokens = set()
        for chunk in chunks:
            context_tokens.update(normalize_tokens(chunk.text + " " + chunk.title))

        overlap = len(query_tokens.intersection(context_tokens)) / max(len(query_tokens), 1)
        top_score = max(c.score for c in chunks) if chunks else 0.0

        in_domain = self.is_in_domain(query, top_score)

        # Composite confidence metric
        confidence = min(
            0.30 * overlap + 0.40 * min(top_score / 2.0, 1.0) + (0.30 if in_domain else 0.0),
            1.0,
        )
        is_grounded = (confidence >= self.confidence_threshold) and in_domain

        return round(confidence, 3), is_grounded

    def validate_answer_groundedness(self, answer: str, chunks: List[DocumentChunk]) -> bool:
        """
        Validates that generated answer is supported by retrieved context chunks.
        Prevents unsupported hallucinations.
        """
        if not answer or not chunks:
            return False

        answer_tokens = set(normalize_tokens(answer))
        if len(answer_tokens) < 5:
            return True

        context_tokens = set()
        for c in chunks:
            context_tokens.update(normalize_tokens(c.text + " " + c.title))

        # Check content overlap ratio
        overlap = len(answer_tokens.intersection(context_tokens)) / max(len(answer_tokens), 1)
        return overlap >= 0.20

    def get_refusal_response(self) -> str:
        """Returns standard domain refusal message."""
        return DOMAIN_REFUSAL_MESSAGE
