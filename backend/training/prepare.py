"""
backend/training/prepare.py
----------------------------------------------------
Instruction dataset compiler for Genkit AI V6.
Converts company domain data into structured instruction tuning format:
User: <Question>
Assistant: <Verified Answer>
Adds out-of-domain negative examples to teach model scope refusal.
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger
from app.rag.grounding import DOMAIN_REFUSAL_MESSAGE

# Out-of-domain negative training samples
OUT_OF_DOMAIN_EXAMPLES = [
    ("What is the capital of France?", DOMAIN_REFUSAL_MESSAGE),
    ("Who won the 2022 World Cup in football?", DOMAIN_REFUSAL_MESSAGE),
    ("What is the weather like in New York today?", DOMAIN_REFUSAL_MESSAGE),
    ("Write a poem about the ocean and mountains.", DOMAIN_REFUSAL_MESSAGE),
    ("How do I cook spaghetti bolognese?", DOMAIN_REFUSAL_MESSAGE),
    ("Explain the theory of general relativity.", DOMAIN_REFUSAL_MESSAGE),
    ("Who is the prime minister of the UK?", DOMAIN_REFUSAL_MESSAGE),
    ("What is the stock price of Apple right now?", DOMAIN_REFUSAL_MESSAGE),
    ("Help me solve this calculus differential equation.", DOMAIN_REFUSAL_MESSAGE),
    ("Translate this paragraph into German.", DOMAIN_REFUSAL_MESSAGE),
]


def build_instruction_corpus() -> List[str]:
    """Compiles all verified domain JSON data into formatted instruction strings."""
    dataset_dir = settings.DATASET_DIR
    corpus: List[str] = []

    if not dataset_dir.exists():
        return corpus

    for json_file in sorted(dataset_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        q = item.get("instruction") or item.get("question") or item.get("title")
                        a = item.get("output") or item.get("answer") or item.get("description") or item.get("content")
                        if q and a and len(str(q).strip()) > 3 and len(str(a).strip()) > 3:
                            corpus.append(f"User: {str(q).strip()}\nAssistant: {str(a).strip()}")
                        elif a and len(str(a).strip()) > 10:
                            corpus.append(str(a).strip())
                    elif isinstance(item, str) and len(item.strip()) > 10:
                        corpus.append(item.strip())

            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, str) and len(v.strip()) > 5:
                        corpus.append(f"User: Tell me about {k.replace('_', ' ')}.\nAssistant: {v.strip()}")
        except Exception as e:
            logger.error(f"Error loading {json_file.name}: {e}")

    # Add negative out-of-domain samples
    for q, a in OUT_OF_DOMAIN_EXAMPLES:
        corpus.append(f"User: {q}\nAssistant: {a}")

    logger.info(f"Compiled {len(corpus):,} instruction training sentences.")
    return corpus
