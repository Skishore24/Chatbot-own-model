def detect_intent(query: str) -> dict:
    """
    Production-level intent detection with scoring + domain guard
    Returns:
    {
        "intent": str,
        "is_out_of_scope": bool,
        "confidence": float
    }
    """

    q = query.lower().strip()

    # ─────────────────────────────────────────────
    # OUT OF SCOPE (STRICT DOMAIN CONTROL)
    # ─────────────────────────────────────────────
    out_of_scope_keywords = [
        "weather", "joke", "recipe", "movie", "song",
        "sports", "cricket", "football", "news",
        "crypto", "stock", "politics", "election",
        "health", "medicine", "advice", "who is elon",
        "who is bill gates", "president"
    ]

    # Allow Genkit-related queries
    if any(w in q for w in out_of_scope_keywords) and "genkit" not in q:
        return {
            "intent": "irrelevant",
            "is_out_of_scope": True,
            "confidence": 1.0
        }

    # ─────────────────────────────────────────────
    # INTENT KEYWORDS (EXPANDED)
    # ─────────────────────────────────────────────
    intents = {
        "pricing": [
            "price", "cost", "budget", "pricing", "charges",
            "how much", "quotation", "estimate", "fee"
        ],
        "services": [
            "service", "services", "offer", "solutions",
            "what do you do", "what you provide", "features"
        ],
        "contact": [
            "contact", "email", "phone", "reach",
            "connect", "call", "message"
        ],
        "about": [
            "about", "company", "genkit", "who are you",
            "information", "details", "history"
        ],
        "project": [
            "project", "build", "develop", "create",
            "make", "design", "website", "app", "chatbot"
        ]
    }

    # ─────────────────────────────────────────────
    # SCORING SYSTEM
    # ─────────────────────────────────────────────
    scores = {intent: 0 for intent in intents}

    for intent, keywords in intents.items():
        for word in keywords:
            if word in q:
                scores[intent] += 1

    # ─────────────────────────────────────────────
    # SELECT BEST INTENT
    # ─────────────────────────────────────────────
    best_intent = max(scores, key=scores.get)
    max_score = scores[best_intent]

    # ─────────────────────────────────────────────
    # CONFIDENCE CALCULATION
    # ─────────────────────────────────────────────
    total_hits = sum(scores.values()) or 1
    confidence = max_score / total_hits

    # If no strong signal → general
    if max_score == 0:
        return {
            "intent": "general",
            "is_out_of_scope": False,
            "confidence": 0.3
        }

    return {
        "intent": best_intent,
        "is_out_of_scope": False,
        "confidence": round(confidence, 2)
    }