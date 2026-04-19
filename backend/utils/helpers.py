import re


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Normalize text (remove extra spaces, lower noise)."""
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip()


# ─────────────────────────────────────────────
# QUERY VALIDATION
# ─────────────────────────────────────────────
def is_valid_query(query: str) -> bool:
    """
    Filters clearly irrelevant or spam queries.
    Keeps system focused on business domain.
    """

    if not query or len(query.strip()) < 2:
        return False

    blocked = [
        "elon musk", "weather", "cricket",
        "movie", "song", "stock", "bitcoin",
        "politics", "news", "recipe", "joke"
    ]

    q = query.lower()

    return not any(b in q for b in blocked)


# ─────────────────────────────────────────────
# NAME EXTRACTION
# ─────────────────────────────────────────────
def extract_name(text: str):
    """
    Extract user name from natural language.
    """
    if not text:
        return None

    patterns = [
        r"my name is (\w+)",
        r"i am (\w+)",
        r"call me (\w+)",
        r"this is (\w+)"
    ]

    t = text.lower()

    for p in patterns:
        match = re.search(p, t)
        if match:
            return match.group(1).capitalize()

    return None


# ─────────────────────────────────────────────
# EMAIL EXTRACTION (🔥 IMPORTANT FOR LEADS)
# ─────────────────────────────────────────────
def extract_email(text: str):
    """Extract email from user message."""
    if not text:
        return None

    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    if match:
        return match.group(0)

    return None


# ─────────────────────────────────────────────
# LEAD DETECTION (UPGRADED)
# ─────────────────────────────────────────────
def detect_lead(text: str) -> bool:
    """
    Detect high-intent business queries.
    Used for lead capture system.
    """

    if not text:
        return False

    t = text.lower()

    keywords = [
        # pricing intent
        "price", "cost", "budget", "how much",
        "quotation", "estimate",

        # hiring intent
        "hire", "developer", "agency", "team",

        # project intent
        "project", "build", "create", "develop",
        "website", "app", "chatbot",

        # contact intent
        "contact", "email", "call", "reach"
    ]

    return any(k in t for k in keywords)


# ─────────────────────────────────────────────
# QUERY TYPE (OPTIONAL FUTURE USE)
# ─────────────────────────────────────────────
def classify_query_type(query: str) -> str:
    """
    Optional helper for future scaling.
    Returns: info | pricing | contact | general
    """

    q = query.lower()

    if any(k in q for k in ["price", "cost", "budget"]):
        return "pricing"

    if any(k in q for k in ["contact", "email", "phone"]):
        return "contact"

    if any(k in q for k in ["service", "offer", "what do you do"]):
        return "info"

    return "general"