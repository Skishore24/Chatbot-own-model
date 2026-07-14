"""
ai/evaluation/metrics.py
----------------------------------------------------
Genkit AI Evaluation Metrics

Provides:
- Exact Match
- BLEU (1-gram + 2-gram)
- Precision
- Recall
- F1 Score
- Relevance
- Grounding
- Hallucination
- Overall Score

Author: Genkit AI
"""

import math
import re
from typing import Dict, List


# ============================================================
# TOKENIZATION
# ============================================================

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


# ============================================================
# EXACT MATCH
# ============================================================

def exact_match(prediction: str, reference: str) -> float:
    return float(
        prediction.strip().lower() ==
        reference.strip().lower()
    )


# ============================================================
# BLEU
# ============================================================

def bleu_score(prediction: str, reference: str) -> float:

    pred = _tokenize(prediction)
    ref = _tokenize(reference)

    if not pred or not ref:
        return 0.0

    # 1 gram

    ref_count = {}

    for w in ref:
        ref_count[w] = ref_count.get(w, 0) + 1

    match1 = 0

    tmp = ref_count.copy()

    for w in pred:

        if tmp.get(w, 0):

            match1 += 1
            tmp[w] -= 1

    p1 = match1 / len(pred)

    # 2 gram

    pred2 = _ngrams(pred, 2)
    ref2 = _ngrams(ref, 2)

    ref_count = {}

    for g in ref2:
        ref_count[g] = ref_count.get(g, 0) + 1

    match2 = 0

    tmp = ref_count.copy()

    for g in pred2:

        if tmp.get(g, 0):

            match2 += 1
            tmp[g] -= 1

    p2 = match2 / len(pred2) if pred2 else 0

    if p1 == 0 or p2 == 0:
        return round(p1, 4)

    bp = min(1.0, len(pred) / len(ref))

    bleu = bp * math.sqrt(p1 * p2)

    return round(bleu, 4)


# ============================================================
# PRECISION
# ============================================================

def precision(prediction: str, reference: str):

    pred = set(_tokenize(prediction))
    ref = set(_tokenize(reference))

    if not pred:
        return 0

    return round(len(pred & ref) / len(pred), 4)


# ============================================================
# RECALL
# ============================================================

def recall(prediction: str, reference: str):

    pred = set(_tokenize(prediction))
    ref = set(_tokenize(reference))

    if not ref:
        return 0

    return round(len(pred & ref) / len(ref), 4)


# ============================================================
# F1 SCORE
# ============================================================

def f1_score(prediction: str, reference: str):

    p = precision(prediction, reference)
    r = recall(prediction, reference)

    if p + r == 0:
        return 0

    return round(
        2 * p * r / (p + r),
        4
    )


# ============================================================
# GROUNDING
# ============================================================

def grounding_score(response: str, context: str):

    if not context:
        return 0

    resp = set(_tokenize(response))
    ctx = set(_tokenize(context))

    if not resp:
        return 0

    return round(
        len(resp & ctx) / len(resp),
        4
    )


# ============================================================
# RELEVANCE
# ============================================================

def relevance_score(query: str, response: str):

    if not query:
        return 0

    q = set(_tokenize(query))
    r = set(_tokenize(response))

    if not q:
        return 0

    return round(
        len(q & r) / len(q),
        4
    )


# ============================================================
# HALLUCINATION
# ============================================================

def hallucination_score(response: str, context: str):

    if not context:
        return 1.0

    resp = set(_tokenize(response))
    ctx = set(_tokenize(context))

    if not resp:
        return 0

    hallucinated = resp - ctx

    return round(
        len(hallucinated) / len(resp),
        4
    )


# ============================================================
# OVERALL
# ============================================================

def overall_score(metrics: Dict[str, float]):

    values = [
        metrics["bleu"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["grounding"],
        metrics["relevance"]
    ]

    return round(sum(values) / len(values), 4)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    prediction: str,
    reference: str,
    query: str = "",
    context: str = ""
):

    metrics = {

        "exact_match":
            exact_match(prediction, reference),

        "bleu":
            bleu_score(prediction, reference),

        "precision":
            precision(prediction, reference),

        "recall":
            recall(prediction, reference),

        "f1":
            f1_score(prediction, reference),

        "grounding":
            grounding_score(prediction, context),

        "relevance":
            relevance_score(query, prediction),

        "hallucination":
            hallucination_score(prediction, context)
    }

    metrics["overall"] = overall_score(metrics)

    return metrics
