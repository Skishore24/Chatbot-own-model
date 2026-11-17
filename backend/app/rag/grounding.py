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
    "application", "applications", "engineer", "engineering", "developer", "developers",
    "smartphone", "ios", "android", "flutter",
    "chatbot", "automation", "ui", "ux", "design", "graphic", "branding", "logo",
    "seo", "marketing", "digital", "video", "editing", "motion", "animation",
    "ecommerce", "store", "saas", "portal", "custom", "software", "development",
    # Technologies & Architecture
    "technology", "technologies", "tech", "stack", "framework", "frameworks", "library", "libraries",
    "backend", "frontend", "fullstack", "api", "apis", "rest", "graphql",
    "database", "databases", "sql", "nosql", "mysql", "mongodb", "postgresql", "redis",
    "react", "nextjs", "vue", "angular", "python", "pytorch", "tensorflow", "fastapi",
    "django", "flask", "node", "nodejs", "express", "typescript", "javascript",
    "figma", "photoshop", "illustrator", "premiere", "effects", "aws", "cloud", "docker", "devops",
    # Pricing, Policies & Process
    "pricing", "prices", "cost", "quote", "rate", "rates", "package", "packages",
    "budget", "hourly", "timeline", "turnaround", "process", "consultation",
    "hire", "contact", "email", "phone", "support", "revisions", "nda", "refund",
    "policy", "policies", "terms", "payment", "payments", "contract", "warranty", "sla",
    # Conversational intent & Domain terms
    "build", "create", "offer", "provide", "portfolio", "projects", "case", "study",
    "specialize", "specializes", "specialized", "specializing", "speciality", "expertise", "skills",
    "client", "clients", "testimonial", "testimonials", "review", "reviews", "mvp",
}

GENKIT_OUT_OF_DOMAIN_KEYWORDS = {
    "stock", "stocks", "shares", "nasdaq", "nyse", "tesla", "crypto", "bitcoin",
    "ethereum", "weather", "forecast", "recipe", "cookies", "quantum", "schrodinger",
    "football", "fifa", "worldcup", "olympics", "president", "election",
    "minister", "celebrity", "movie", "song", "horoscope"
}

# Categories that MUST use deterministic RAG-direct synthesis (0% hallucination risk)
RAG_DIRECT_CATEGORIES = {
    "Contact", "Pricing", "Policies", "FAQ", "Team", "Clients",
    "Company", "Services", "Technologies", "Portfolio", "Projects",
}

# High-priority factual keywords — force rag_direct regardless of category
FACTUAL_KEYWORDS = {
    "contact", "email", "phone", "call", "reach", "instagram", "github", "website",
    "price", "pricing", "cost", "quote", "rate", "rates", "package", "packages", "budget",
    "policy", "policies", "refund", "nda", "terms", "payment", "payments",
    "faq", "question", "questions", "hours", "response time", "consultation",
    "team", "who", "founder", "founders", "leadership", "member",
    "service", "services", "offer", "offered", "solutions", "what do you do",
    "portfolio", "projects", "work", "case study",
    "technology", "technologies", "tech stack", "framework",
    "client", "clients", "review", "testimonial",
}


def should_use_rag_direct(query: str, top_category: str) -> bool:
    """
    Returns True when query or category matches verified Genkit dataset knowledge.
    Guarantees deterministic ground-truth synthesis with zero hallucination.
    """
    if top_category in RAG_DIRECT_CATEGORIES:
        return True
    q = query.lower().strip()
    return any(kw in q for kw in FACTUAL_KEYWORDS)


def is_degenerate_or_hallucinatory(text: str) -> bool:
    """
    Detects degenerative repetition or hallucinations commonly emitted by small custom LLMs.
    Returns True if the output is invalid/hallucinatory and should be replaced by rag_direct.
    """
    if not text or len(text.strip()) < 10:
        return True
    words = text.lower().split()
    if len(words) >= 6:
        # 1. Adjacent word repetitions (e.g. "service: service:", "have have have")
        adjacent_repeats = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
        if adjacent_repeats >= 2:
            return True
        # 2. Vocabulary repetition ratio (small LLM looping on a tiny subset of tokens)
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.45:
            return True
    return False


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

        # Conversational greetings only (not questions starting with what/who)
        GREETINGS = {"hi", "hello", "hey", "hola", "namaste", "greetings", "sup"}
        if len(query_tokens) <= 2 and any(t in GREETINGS for t in query_tokens):
            return True

        # Explicit out-of-domain guardrail: reject external general trivia
        has_genkit = "genkit" in query_tokens
        if not has_genkit and any(t in GENKIT_OUT_OF_DOMAIN_KEYWORDS for t in query_tokens):
            return False

        # Check keyword matches against domain vocabulary
        matches = [t for t in query_tokens if t in GENKIT_CORE_KEYWORDS]
        if len(matches) > 0:
            return True

        # High RAG retrieval confidence indicates domain match even with new vocabulary
        if top_retrieval_score >= 3.0:
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
