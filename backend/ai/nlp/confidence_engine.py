"""
ai/nlp/confidence_engine.py
----------------------------------------------------
Genkit AI - Confidence Evaluation Engine

Computes overall confidence based on:
1. Intent Classifier Score (Weight: 20%)
2. Retriever Similarity Score (Weight: 35%)
3. Transformer Generation Score (Weight: 45%)

Triggers fallback refusal if confidence drops below a threshold.

Author : Genkit AI
"""

import logging
from typing import List, Dict

logger = logging.getLogger("Genkit AI")

# Fallback message
LOW_CONFIDENCE_FALLBACK = (
    "I don't have enough information about this topic within Genkit's knowledge base. "
    "Could you please ask about our services, pricing, portfolio, support, or contact info?"
)

class ConfidenceEngine:
    """
    Computes overall confidence for a generated answer.
    """

    def __init__(self, threshold: float = 0.40) -> None:
        self.threshold = threshold

    def compute(
        self,
        intent_score: float,
        retrieved_docs: List[Dict],
        generation_score: float
    ) -> Dict:
        """
        Computes retriever, intent, generation, and overall confidence scores.
        """
        # 1. Retriever confidence (derived from top doc score)
        if retrieved_docs:
            max_similarity = max(doc.get("score", 0.0) for doc in retrieved_docs)
            # Map score range [0.0 - 1.5] (due to boosts) to [0.0 - 1.0]
            retriever_conf = min(1.0, max_similarity / 1.2)
        else:
            retriever_conf = 0.0

        # 2. Intent confidence
        intent_conf = min(1.0, max(0.0, intent_score))

        # 3. Generation confidence (average token prob)
        generation_conf = min(1.0, max(0.0, generation_score))

        # 4. Weighted Overall Confidence
        # Higher weight given to text generation likelihood and RAG retrieval match
        overall_score = (
            0.20 * intent_conf +
            0.35 * retriever_conf +
            0.45 * generation_conf
        )

        overall_score = round(overall_score, 4)
        is_low = overall_score < self.threshold

        logger.info(
            f"[Confidence] Overall={overall_score:.3f} | "
            f"Intent={intent_conf:.3f} | RAG={retriever_conf:.3f} | Gen={generation_conf:.3f} | Low={is_low}"
        )

        return {
            "overall": overall_score,
            "intent_confidence": round(intent_conf, 3),
            "retriever_confidence": round(retriever_conf, 3),
            "generation_confidence": round(generation_conf, 3),
            "low_confidence": is_low,
            "fallback_message": LOW_CONFIDENCE_FALLBACK
        }

# Singleton
confidence_engine = ConfidenceEngine()

__all__ = ["ConfidenceEngine", "confidence_engine", "LOW_CONFIDENCE_FALLBACK"]
