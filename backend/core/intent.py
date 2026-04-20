def detect_intent(query: str) -> dict:
    """
    Production-level intent detection with:
    - scoring
    - priority handling
    - domain filtering
    - fallback safety
    """

    q = query.lower().strip()

    # ─────────────────────────────────────────────
    # STRICT OUT-OF-SCOPE FILTER
    # ─────────────────────────────────────────────
    out_of_scope_keywords = [
        "weather", "joke", "recipe", "movie", "song",
        "sports", "cricket", "football", "news",
        "crypto", "stock", "politics", "election",
        "health", "medicine", "doctor", "advice",
        "who is elon", "who is bill gates",
        "president", "prime minister"
    ]

    # HARD BLOCK unless Genkit mentioned
    if any(w in q for w in out_of_scope_keywords):
        if "genkit" not in q:
            return {
                "intent": "irrelevant",
                "is_out_of_scope": True,
                "confidence": 1.0
            }

    # ─────────────────────────────────────────────
    # INTENT KEYWORDS
    # ─────────────────────────────────────────────
    intents = {
        "pricing": [
            "price", "cost", "budget", "pricing",
            "charges", "how much", "quotation",
            "estimate", "fee", "rate"
        ],
        "services": [
            "service", "services", "offer",
            "solutions", "features",
            "what do you do", "what you provide"
        ],
        "contact": [
            "contact", "email", "phone",
            "reach", "connect", "call",
            "message", "address"
        ],
        "about": [
            "about", "company", "genkit",
            "who are you", "details",
            "information", "history"
        ],
        "project": [
            "project", "build", "develop",
            "create", "make", "design",
            "website", "app", "chatbot",
            "portfolio"
        ]
    }

    # ─────────────────────────────────────────────
    # PRIORITY ORDER (VERY IMPORTANT)
    # ─────────────────────────────────────────────
    priority = ["pricing", "contact", "project", "services", "about"]

    # ─────────────────────────────────────────────
    # SCORING SYSTEM (IMPROVED)
    # ─────────────────────────────────────────────
    scores = {intent: 0 for intent in intents}

    for intent, keywords in intents.items():
        for word in keywords:
            if word in q:
                # phrase gets more weight
                if len(word.split()) > 1:
                    scores[intent] += 2
                else:
                    scores[intent] += 1

    # ─────────────────────────────────────────────
    # BOOST FOR GENKIT CONTEXT
    # ─────────────────────────────────────────────
    if "genkit" in q:
        scores["about"] += 1
        scores["services"] += 1

    # ─────────────────────────────────────────────
    # SELECT BEST INTENT WITH PRIORITY
    # ─────────────────────────────────────────────
    max_score = max(scores.values())

    if max_score == 0:
        return {
            "intent": "general",
            "is_out_of_scope": False,
            "confidence": 0.3
        }

    # pick all with same score
    candidates = [k for k, v in scores.items() if v == max_score]

    # apply priority
    for p in priority:
        if p in candidates:
            best_intent = p
            break

    # ─────────────────────────────────────────────
    # CONFIDENCE
    # ─────────────────────────────────────────────
    total_hits = sum(scores.values()) or 1
    confidence = round(max_score / total_hits, 2)

    # ─────────────────────────────────────────────
    # FINAL SAFETY CHECK
    # ─────────────────────────────────────────────
    if confidence < 0.2:
        return {
            "intent": "general",
            "is_out_of_scope": False,
            "confidence": confidence
        }

    return {
        "intent": best_intent,
        "is_out_of_scope": False,
        "confidence": confidence
    }