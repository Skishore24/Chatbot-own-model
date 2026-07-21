"""
ai/nlp/response_validator.py
----------------------------------------------------
Genkit AI - Output Response Validator

Performs validation checks on the generated text:
1. Anti-hallucination: ensures mentioned tech and pricing numbers exist in retrieved context.
2. Repetition: detects sentence loops or repeated word patterns.
3. Length: flags responses that are too short or empty.
4. Off-topic: prevents out-of-scope leakages.

Author : Genkit AI
"""

import re
import logging
from typing import List, Dict, Tuple
from ai.nlp.entity_extractor import entity_extractor

logger = logging.getLogger("Genkit AI")

# Out of scope leak warning keywords
OOS_KEYWORDS = {"weather", "crypto", "sports", "recipe", "election", "politics", "celeb", "fifa", "cricket"}

class ResponseValidator:
    """
    Validates model output to ensure factual accuracy and safe guardrails.
    """

    def validate(self, generated_text: str, retrieved_docs: List[Dict], query: str) -> Tuple[bool, str]:
        """
        Validates the generated text.

        Returns:
            (success: bool, reason: str)
        """
        if not generated_text or len(generated_text.strip()) < 15:
            return False, "Response is too short or empty."

        # Merge all retrieved document texts for verification
        context_parts = []
        for doc in retrieved_docs:
            for k, v in doc.items():
                if isinstance(v, str):
                    context_parts.append(v.lower())
        context_text = "\n".join(context_parts)
        query_lower = query.lower()

        # 1. Repetition Check
        sentences = [s.strip() for s in re.split(r'[.!?]', generated_text) if len(s.strip()) > 5]
        if len(sentences) > 3:
            # Check for high fraction of duplicated sentences
            seen = set()
            duplicates = 0
            for s in sentences:
                s_norm = re.sub(r"[^\w]", "", s.lower())
                if s_norm in seen:
                    duplicates += 1
                seen.add(s_norm)
            if duplicates / len(sentences) > 0.25:
                return False, "Response contains excessive sentence duplication."

        # 2. Anti-Hallucination: Price Check
        # Match dollar values e.g. $500, 500 USD, $1,500
        generated_prices = re.findall(r"\$?\b\d{3,5}\b", generated_text)
        for price in generated_prices:
            # Normalize to digit string e.g. "500"
            price_digits = re.sub(r"[^\d]", "", price)
            if price_digits not in context_text and price_digits not in query_lower:
                logger.warning(f"[Validator] Hallucinated price detected: '{price}' not in context.")
                return False, f"Hallucinated price value '{price}' detected."

        # 3. Anti-Hallucination: Technology Check
        entities = entity_extractor.extract(generated_text)
        for tech in entities["technologies"]:
            tech_lower = tech.lower()
            if tech_lower not in context_text and tech_lower not in query_lower:
                # Allow general description reference, but flag if totally unrelated
                logger.warning(f"[Validator] Hallucinated tech detected: '{tech}' not in context.")
                return False, f"Hallucinated technology '{tech}' detected."

        # 4. Off-Topic Safeguard
        words = set(re.findall(r"\b\w+\b", generated_text.lower()))
        oos_hits = words & OOS_KEYWORDS
        if oos_hits:
            logger.warning(f"[Validator] Off-topic keywords detected: {oos_hits}")
            return False, "Response contains out-of-scope subjects."

        return True, "Success"

# Singleton
response_validator = ResponseValidator()

__all__ = ["ResponseValidator", "response_validator"]
