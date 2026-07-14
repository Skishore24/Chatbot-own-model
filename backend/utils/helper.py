"""
utils/helper.py
----------------------------------------------------
Genkit AI - Production Helper Utilities
Features
--------
• Text Cleaning
• Query Validation
• Email Extraction
• Phone Extraction
• URL Extraction
• Name Detection
• Greeting Detection
• Spam Detection
• Business Lead Detection
• Intent Detection
• Grounding Helpers
Author : Genkit AI
"""
import re
import string
from typing import List, Dict, Optional
# ============================================================
# CONSTANTS
# ============================================================
MAX_RESPONSE_WORDS = 80
MAX_CONTEXT_DOCS = 5
MIN_QUERY_LENGTH = 2
EMAIL_REGEX = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
PHONE_REGEX = re.compile(
    r"(\+?\d[\d\s\-]{8,15})"
)
URL_REGEX = re.compile(
    r"(https?://\S+|www\.\S+)"
)
NAME_REGEX = re.compile(
    r"(?:my name is|i am|call me|this is)\s+([A-Za-z ]+)",
    re.IGNORECASE
)
# ============================================================
# STOP WORDS
# ============================================================
STOP_WORDS = {
    "a","an","the","is","are","was","were",
    "to","of","for","in","on","at","by",
    "this","that","these","those",
    "my","your","their","our",
    "can","could","would","should",
    "please","tell","give","about"
}
# ============================================================
# COMPANY KEYWORDS
# ============================================================
GENKIT_KEYWORDS = {
    "genkit",
    "website",
    "web",
    "chatbot",
    "ai",
    "automation",
    "design",
    "branding",
    "seo",
    "marketing",
    "video",
    "editing",
    "mobile",
    "android",
    "ios",
    "software",
    "application",
    "portfolio",
    "hosting",
    "domain",
    "figma",
    "python",
    "react",
    "node",
    "mysql",
    "mongodb"
}
# ============================================================
# TEXT CLEANING
# ============================================================
def clean_query(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )
    text = re.sub(
        URL_REGEX,
        " ",
        text
    )
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    return text.strip()

def normalize_text(text: str) -> str:
    text = clean_query(text)
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )
    return text

# ============================================================
# RESPONSE CLEANING
# ============================================================
def clean_response(
    text: str,
    max_words: int = MAX_RESPONSE_WORDS
):
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(
            words[:max_words]
        )
    return text.strip()
# ============================================================
# EXTRACTION HELPERS
# ============================================================
def extract_email(text: str) -> Optional[str]:
    match = EMAIL_REGEX.search(text)
    if match:
        return match.group()
    return None

def extract_phone(text: str) -> Optional[str]:
    match = PHONE_REGEX.search(text)
    if match:
        return match.group(1)
    return None

def extract_url(text: str) -> Optional[str]:
    match = URL_REGEX.search(text)
    if match:
        return match.group()
    return None

def extract_name(text: str) -> Optional[str]:
    match = NAME_REGEX.search(text)
    if match:
        return match.group(1).strip().title()
    return None
# ============================================================
# GREETING DETECTION
# ============================================================
_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
    "welcome"
}

_GOODBYE = {
    "bye",
    "goodbye",
    "see you",
    "take care",
    "see ya",
    "later"
}

_THANKS = {
    "thanks",
    "thank you",
    "thankyou",
    "thx",
    "ty"
}

def is_greeting(text: str) -> bool:
    text = clean_query(text)
    return any(
        g in text
        for g in _GREETINGS
    )

def is_goodbye(text: str) -> bool:
    text = clean_query(text)
    return any(
        g in text
        for g in _GOODBYE
    )

def is_thanks(text: str) -> bool:
    text = clean_query(text)
    return any(
        g in text
        for g in _THANKS
    )

# ============================================================
# QUERY VALIDATION
# ============================================================
_BLOCKED_TERMS = {
    "crack",
    "hack facebook",
    "hack instagram",
    "porn",
    "adult",
    "torrent",
    "pirated",
    "bitcoin",
    "lottery",
    "casino"
}

def is_valid_query(query: str) -> bool:
    if not query:
        return False
    query = clean_query(query)
    if len(query) < MIN_QUERY_LENGTH:
        return False
    for word in _BLOCKED_TERMS:
        if word in query:
            return False
    return True

# ============================================================
# SPAM DETECTION
# ============================================================
def is_spam(text: str) -> bool:
    if not text:
        return True
    text = clean_query(text)
    if len(text) < 2:
        return True
    if len(set(text)) <= 2:
        return True
    repeated = re.search(
        r"(.)\1{6,}",
        text
    )
    if repeated:
        return True
    symbols = sum(
        1
        for c in text
        if not c.isalnum()
    )
    if symbols > len(text) * 0.60:
        return True
    return False

# ============================================================
# BUSINESS KEYWORD SCORE
# ============================================================
def keyword_score(text: str) -> int:
    words = normalize_text(text).split()
    score = 0
    for word in words:
        if word in GENKIT_KEYWORDS:
            score += 1
    return score

def contains_company_keyword(text: str) -> bool:
    return keyword_score(text) > 0

# ============================================================
# LEAD DETECTION
# ============================================================
LEAD_WORDS = {
    "hire",
    "price",
    "pricing",
    "quotation",
    "quote",
    "budget",
    "website",
    "web",
    "chatbot",
    "software",
    "application",
    "mobile",
    "android",
    "ios",
    "seo",
    "marketing",
    "contact",
    "call",
    "email",
    "project",
    "startup",
    "business"
}

def detect_lead(text: str) -> bool:
    words = normalize_text(text).split()
    score = 0
    for word in words:
        if word in LEAD_WORDS:
            score += 1
    return score >= 2
# ============================================================
# INTENT DETECTION
# ============================================================
INTENT_KEYWORDS = {
    "about": [
        "about",
        "company",
        "genkit",
        "who are you",
        "history",
        "mission",
        "vision",
        "team"
    ],
    "services": [
        "service",
        "services",
        "solution",
        "offer",
        "provide",
        "website",
        "design",
        "video",
        "branding",
        "seo"
    ],
    "pricing": [
        "price",
        "pricing",
        "cost",
        "budget",
        "quotation",
        "quote",
        "estimate",
        "fee"
    ],
    "contact": [
        "contact",
        "email",
        "phone",
        "call",
        "reach",
        "address",
        "location"
    ],
    "support": [
        "support",
        "issue",
        "problem",
        "bug",
        "error",
        "help",
        "login"
    ],
    "career": [
        "career",
        "job",
        "internship",
        "vacancy",
        "work",
        "join"
    ],
    "portfolio": [
        "portfolio",
        "projects",
        "clients",
        "work",
        "examples"
    ]
}

OUT_OF_SCOPE = {
    "weather",
    "cricket",
    "football",
    "movie",
    "song",
    "bitcoin",
    "crypto",
    "politics",
    "president",
    "prime minister",
    "recipe",
    "medicine",
    "doctor",
    "ipl",
    "stock"
}

def detect_intent(query: str):
    query = clean_query(query)
    scores = {}
    for intent in INTENT_KEYWORDS:
        scores[intent] = 0
        for keyword in INTENT_KEYWORDS[intent]:
            if keyword in query:
                if " " in keyword:
                    scores[intent] += 2
                else:
                    scores[intent] += 1
    best = "general"
    highest = 0
    for intent, score in scores.items():
        if score > highest:
            highest = score
            best = intent
    confidence = 0.30
    if highest > 0:
        total = sum(scores.values())
        confidence = round(
            highest / max(total, 1),
            2
        )
    return {
        "intent": best,
        "confidence": confidence,
        "scores": scores
    }

# ============================================================
# OUT OF SCOPE
# ============================================================
def is_out_of_scope(text: str):
    text = clean_query(text)
    if "genkit" in text:
        return False
    for word in OUT_OF_SCOPE:
        if word in text:
            return True
    return False

# ============================================================
# CONTEXT BUILDER
# ============================================================
def build_context(
    documents,
    max_docs=5
):
    if not documents:
        return ""
    context = []
    for doc in documents[:max_docs]:
        if isinstance(doc, dict):
            context.append(
                doc.get(
                    "text",
                    ""
                )
            )
        else:
            context.append(
                str(doc)
            )
    return "\n\n".join(context)

# ============================================================
# MEMORY TRIMMER
# ============================================================
def trim_memory(
    history,
    max_messages=5
):
    if not history:
        return []
    return history[-max_messages:]

# ============================================================
# GROUNDING CHECK
# ============================================================
def is_grounded(
    response,
    context,
    minimum_overlap=3
):
    if not response:
        return False
    response_words = set(
        normalize_text(
            response
        ).split()
    )
    context_words = set(
        normalize_text(
            context
        ).split()
    )
    overlap = len(
        response_words & context_words
    )
    return overlap >= minimum_overlap

# ============================================================
# RESPONSE FORMATTER
# ============================================================
def format_response(
    text,
    max_lines=6
):
    if not text:
        return ""
    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]
    lines = lines[:max_lines]
    return "\n".join(lines)
# ============================================================
# EMAIL VALIDATION
# ============================================================
def is_valid_email(email: str) -> bool:
    """
    Validate email address.
    """
    if not email:
        return False
    return EMAIL_REGEX.fullmatch(email.strip()) is not None

# ============================================================
# PHONE VALIDATION
# ============================================================
def is_valid_phone(phone: str) -> bool:
    """
    Validate phone number.
    """
    if not phone:
        return False
    phone = phone.replace(" ", "").replace("-", "")
    return PHONE_REGEX.fullmatch(phone) is not None

# ============================================================
# BUSINESS LEAD SCORE
# ============================================================
def lead_score(text: str) -> int:
    """
    Return lead score (0-100).
    """
    text = normalize_text(text)
    score = 0
    for keyword in LEAD_WORDS:
        if keyword in text:
            score += 10
    if extract_email(text):
        score += 20
    if extract_phone(text):
        score += 20
    return min(score, 100)

# ============================================================
# COMPANY QUERY
# ============================================================
def is_company_query(text: str) -> bool:
    """
    Check whether query belongs to Genkit.
    """
    text = normalize_text(text)
    return keyword_score(text) > 0

# ============================================================
# QUERY SUMMARY
# ============================================================
def analyze_query(text: str) -> Dict:
    return {
        "clean_query": clean_query(text),
        "intent": detect_intent(text),
        "lead": detect_lead(text),
        "lead_score": lead_score(text),
        "company_query": is_company_query(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "name": extract_name(text),
        "url": extract_url(text),
        "spam": is_spam(text),
        "valid": is_valid_query(text),
        "greeting": is_greeting(text),
        "thanks": is_thanks(text),
        "goodbye": is_goodbye(text),
        "out_of_scope": is_out_of_scope(text)
    }

# ============================================================
# DEFAULT RESPONSES
# ============================================================
DEFAULT_RESPONSES = {
    "greeting":
        "Hello! Welcome to Genkit AI. How can I help you today?",
    "thanks":
        "You're welcome! If you need any information about Genkit services, feel free to ask.",
    "goodbye":
        "Thank you for contacting Genkit. Have a great day!",
    "out_of_scope":
        "I can assist only with Genkit services, products and business information."
}

# ============================================================
# EXPORTS
# ============================================================
__all__ = [
    "clean_query",
    "normalize_text",
    "clean_response",
    "extract_email",
    "extract_phone",
    "extract_url",
    "extract_name",
    "is_greeting",
    "is_goodbye",
    "is_thanks",
    "is_valid_query",
    "is_valid_email",
    "is_valid_phone",
    "is_spam",
    "detect_intent",
    "detect_lead",
    "lead_score",
    "contains_company_keyword",
    "keyword_score",
    "build_context",
    "trim_memory",
    "format_response",
    "is_grounded",
    "is_out_of_scope",
    "is_company_query",
    "analyze_query",
    "DEFAULT_RESPONSES",
]
