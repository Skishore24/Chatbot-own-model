"""
ai/nlp/intent_classifier.py
----------------------------------------------------
Genkit AI - Intent Classifier (v4.0)

Classifies the user's intent from their query using a trained
scikit-learn classifier, falling back to keyword scoring if not trained.

Intents:
-------
greeting, pricing, services, technology, contact, portfolio, project, support, out_of_domain

Author : Genkit AI
"""

import os
import re
import joblib
import logging
from typing import Dict, List, Tuple
from config import MODEL_DIR

logger = logging.getLogger("Genkit AI")

# ============================================================
# INTENT KEYWORD MAP (Fallback)
# ============================================================
INTENT_MAP: Dict[str, List[Tuple[str, float]]] = {
    "greeting": [
        ("hi", 2.0), ("hello", 2.0), ("hey", 2.0), ("good morning", 2.5),
        ("good afternoon", 2.5), ("good evening", 2.5), ("greetings", 2.0),
    ],
    "contact": [
        ("contact", 3.0), ("email", 2.0), ("phone", 2.0), ("reach", 2.5),
        ("instagram", 1.5), ("website", 1.5), ("form", 1.5)
    ],
    "pricing": [
        ("price", 2.0), ("pricing", 2.0), ("cost", 2.0), ("budget", 2.0),
        ("quote", 2.0), ("fee", 2.0), ("rate", 1.5), ("hourly", 2.0)
    ],
    "services": [
        ("service", 2.0), ("services", 2.0), ("solutions", 1.5),
        ("offer", 1.5), ("provide", 1.5), ("build", 1.5), ("develop", 1.5)
    ],
    "technology": [
        ("tech", 2.0), ("technology", 2.0), ("python", 2.0), ("react", 2.0),
        ("fastapi", 2.0), ("node", 1.5), ("mysql", 1.5), ("mongodb", 1.5),
        ("rag", 3.0), ("llm", 3.0), ("gpt", 2.5), ("model", 2.0), ("transformer", 2.0)
    ],
    "project": [
        ("shopsmart", 3.0), ("lexibot", 3.0), ("zenbites", 3.0), ("techstream", 3.0),
        ("project", 2.0), ("projects", 2.0), ("built", 1.5)
    ],
    "portfolio": [
        ("portfolio", 3.0), ("work", 2.0), ("previous", 2.0), ("past", 2.0),
        ("examples", 2.5), ("case study", 3.0)
    ],
    "about": [
        ("about", 3.0), ("company", 2.5), ("genkit", 2.0), ("who are you", 3.0),
        ("history", 2.5), ("mission", 2.5), ("vision", 2.5), ("founder", 3.0), ("founders", 3.0)
    ],
    "support": [
        ("support", 3.0), ("maintenance", 2.5), ("revisions", 2.0), ("refund", 2.0),
        ("nda", 2.0), ("policy", 2.0), ("help", 1.5)
    ]
}

# Default responses for quick intents
INTENT_QUICK_RESPONSES: Dict[str, str] = {
    "greeting": (
        "Hello! Welcome to Genkit AI. I am here to help you with any questions "
        "about Genkit's digital services and company. What would you like to know?"
    ),
    "out_of_domain": (
        "I am Genkit AI, designed exclusively to assist with questions "
        "about Genkit's digital services, company, and team. "
        "I am not able to help with this topic. "
        "Please feel free to ask me anything about Genkit!"
    )
}

class IntentClassifier:
    """
    ML-based Intent Classifier with Keyword fallback.
    """

    def __init__(self) -> None:
        self.intent_map = INTENT_MAP
        self.clf = None
        self.vectorizer = None
        self.load_model()

    def load_model(self) -> None:
        """Load trained scikit-learn model if available."""
        clf_path = os.path.join(MODEL_DIR, "intent_classifier.joblib")
        vec_path = os.path.join(MODEL_DIR, "intent_vectorizer.joblib")
        if os.path.exists(clf_path) and os.path.exists(vec_path):
            try:
                self.clf = joblib.load(clf_path)
                self.vectorizer = joblib.load(vec_path)
                logger.info("[IntentClassifier] Successfully loaded ML intent model.")
            except Exception as e:
                logger.error(f"[IntentClassifier] Failed to load model files: {e}")

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _score_fallback(self, text: str) -> Tuple[str, float]:
        """Keyword-matching scoring fallback."""
        scores = {intent: 0.0 for intent in self.intent_map}
        normalized = self._normalize(text)
        words = set(re.findall(r"\b\w+\b", normalized))
        
        for intent, keywords in self.intent_map.items():
            for kw, weight in keywords:
                if " " in kw:
                    if kw in normalized:
                        scores[intent] += weight
                else:
                    if kw in words:
                        scores[intent] += weight
                        
        best_intent = "out_of_domain"
        best_score = 0.0
        for intent, score in scores.items():
            if score > best_score:
                best_score = score
                best_intent = intent
                
        total_score = sum(scores.values())
        confidence = round(best_score / total_score, 3) if total_score > 0 else 0.40
        return best_intent, min(confidence, 0.98)

    def classify(self, query: str) -> Dict:
        """
        Classify query intent using either high-precision keyword heuristics or ML model.
        """
        normalized = self._normalize(query)
        
        # Check keyword heuristic for high-precision explicit matches (e.g. founders, pricing)
        fb_intent, fb_conf = self._score_fallback(query)
        if fb_conf >= 0.70 and fb_intent != "out_of_domain":
            return {
                "intent": fb_intent,
                "confidence": fb_conf,
                "scores": {},
                "is_quick": fb_intent in INTENT_QUICK_RESPONSES and fb_conf >= 0.60,
                "quick_response": INTENT_QUICK_RESPONSES.get(fb_intent, ""),
            }

        # If ML model is loaded, use it
        if self.clf is not None and self.vectorizer is not None:
            try:
                X = self.vectorizer.transform([normalized])
                pred_intent = self.clf.predict(X)[0]
                
                # Check for predict_proba availability
                if hasattr(self.clf, "predict_proba"):
                    probs = self.clf.predict_proba(X)[0]
                    classes = self.clf.classes_
                    confidence = float(max(probs))
                    scores = {classes[i]: float(probs[i]) for i in range(len(classes))}
                else:
                    confidence = 0.90
                    scores = {pred_intent: 0.90}
                    
                is_quick = pred_intent in INTENT_QUICK_RESPONSES and confidence >= 0.70
                return {
                    "intent": pred_intent,
                    "confidence": confidence,
                    "scores": scores,
                    "is_quick": is_quick,
                    "quick_response": INTENT_QUICK_RESPONSES.get(pred_intent, ""),
                }
            except Exception as e:
                logger.warning(f"[IntentClassifier] ML prediction failed: {e}. Using fallback.")
                
        return {
            "intent": fb_intent,
            "confidence": fb_conf,
            "scores": {},
            "is_quick": fb_intent in INTENT_QUICK_RESPONSES and fb_conf >= 0.60,
            "quick_response": INTENT_QUICK_RESPONSES.get(fb_intent, ""),
        }

    def get_intent(self, query: str) -> str:
        return self.classify(query)["intent"]

# Singleton
intent_classifier = IntentClassifier()

__all__ = ["IntentClassifier", "intent_classifier", "INTENT_QUICK_RESPONSES"]
