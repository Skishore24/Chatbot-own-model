"""
backend/app/rag/grounding.py
----------------------------------------------------
Grounding Validator & Domain Refusal Guard for Genkit AI V6.
Ensures responses are strictly grounded in Genkit company knowledge and rejects out-of-domain queries.
"""

from typing import List, Tuple
from app.rag.chunker import DocumentChunk
from app.rag.index import normalize_tokens

DOMAIN_REFUSAL_MESSAGE = (
    "I can help with Genkit's company, services, projects, technologies, pricing, "
    "and contact information. I don't have verified information about that topic."
)

GENKIT_CORE_KEYWORDS = {
    "genkit", "company", "service", "services", "web", "website", "mobile", "app", "apps",
    "ai", "ml", "artificial", "intelligence", "machine", "learning", "ui", "ux", "design",
    "price", "pricing", "cost", "quote", "contact", "email", "phone", "address", "founder",
    "founders", "team", "project", "projects", "portfolio", "technology", "technologies",
    "stack", "react", "python", "pytorch", "fastapi", "flutter", "nodejs", "aws", "cloud",
    "client", "clients", "hire", "consultation", "timeline", "process", "faq", "policy",
    "kishore", "surya", "developer", "development", "build", "create", "support",
}


class GroundingValidator:
    """Validates domain relevance, computes groundedness, and triggers refusal."""

    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold

    def is_in_domain(self, query: str) -> bool:
        """Determines if the user query is relevant to Genkit company scope."""
        query_tokens = normalize_tokens(query)
        if not query_tokens:
            return True

        # Check keyword intersection with domain dictionary
        matches = [t for t in query_tokens if t in GENKIT_CORE_KEYWORDS]
        # Allow conversational greetings / short queries
        if len(query_tokens) <= 3 and any(t in {"hi", "hello", "hey", "help", "who", "what"} for t in query_tokens):
            return True

        return len(matches) > 0

    def compute_grounding_score(self, query: str, chunks: List[DocumentChunk]) -> Tuple[float, bool]:
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

        # Overlap fraction
        overlap = len(query_tokens.intersection(context_tokens)) / max(len(query_tokens), 1)
        top_score = max(c.score for c in chunks) if chunks else 0.0

        # Composite confidence metric
        confidence = min(0.30 * overlap + 0.40 * min(top_score / 2.0, 1.0) + (0.30 if self.is_in_domain(query) else 0.0), 1.0)
        is_grounded = (confidence >= self.confidence_threshold) and self.is_in_domain(query)

        return round(confidence, 3), is_grounded

    def get_refusal_response(self) -> str:
        """Returns standard domain refusal message."""
        return DOMAIN_REFUSAL_MESSAGE
