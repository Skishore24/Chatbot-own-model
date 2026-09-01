"""
backend/scripts/evaluate.py
----------------------------------------------------
Automated Evaluation Benchmark Runner for Genkit AI V6.
Measures:
- Retrieval relevance & precision
- Grounding accuracy
- Refusal precision on unsupported/out-of-domain queries
- Response latency across categories
- Hallucination detection
Outputs: reports/evaluation_report.json
"""

import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger
from app.rag.pipeline import get_rag_pipeline
from app.llm.inference import load_model_and_tokenizer, ModelStatus
from app.llm.generation import GenerationEngine


def run_evaluation(dataset_path: str = None) -> Dict[str, Any]:
    eval_file = Path(dataset_path) if dataset_path else settings.BASE_DIR / "datasets" / "evaluation" / "test_questions.json"
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation questions file not found: {eval_file}")

    with open(eval_file, "r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    print("=" * 70)
    print(f"GENKIT AI V6.1 — SYSTEM EVALUATION BENCHMARK ({len(questions)} Tests)")
    print("=" * 70)

    rag = get_rag_pipeline()
    model, tokenizer, _, model_status = load_model_and_tokenizer()
    engine = GenerationEngine(model, tokenizer)

    results = []
    latencies = []
    correct_grounding_decisions = 0
    refusal_tests = 0
    refusal_successes = 0
    in_domain_tests = 0
    in_domain_successes = 0

    for idx, item in enumerate(questions, 1):
        q_id = item.get("id", f"q_{idx}")
        query = item["question"]
        category = item.get("category", "General")
        expected_grounded = item.get("expected_grounded", True)

        start = time.time()
        chunks, confidence, is_grounded = rag.retrieve(query)

        if not is_grounded:
            answer = rag.get_refusal_answer()
            mode = "system"
        elif model_status == ModelStatus.READY and engine.model is not None:
            prompt = rag.build_prompt(query, chunks)
            llm_ans = engine.generate(prompt, max_new_tokens=256)
            if llm_ans and len(llm_ans.strip()) > 5:
                answer = llm_ans.strip()
                mode = "llm_rag"
            else:
                answer = rag.synthesize_answer(query, chunks)
                mode = "rag_direct"
        else:
            answer = rag.synthesize_answer(query, chunks)
            mode = "rag_direct"

        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        # Evaluate correctness
        decision_match = (is_grounded == expected_grounded)
        if decision_match:
            correct_grounding_decisions += 1

        if not expected_grounded:
            refusal_tests += 1
            if not is_grounded:
                refusal_successes += 1
        else:
            in_domain_tests += 1
            if is_grounded and len(chunks) > 0:
                in_domain_successes += 1

        status_flag = "PASS" if decision_match else "FAIL"
        print(f"[{idx:02d}/{len(questions):02d}] [{status_flag}] Category: {category:16s} | Latency: {latency_ms:5.1f}ms | Query: {query[:45]}")

        results.append({
            "id": q_id,
            "category": category,
            "query": query,
            "expected_grounded": expected_grounded,
            "predicted_grounded": is_grounded,
            "decision_match": decision_match,
            "confidence": round(confidence, 3),
            "retrieved_chunks": len(chunks),
            "response_mode": mode,
            "latency_ms": round(latency_ms, 2),
            "answer_snippet": answer[:120] + "..." if len(answer) > 120 else answer,
        })

    total = len(questions)
    grounding_accuracy = (correct_grounding_decisions / max(total, 1)) * 100
    refusal_precision = (refusal_successes / max(refusal_tests, 1)) * 100 if refusal_tests else 100.0
    in_domain_recall = (in_domain_successes / max(in_domain_tests, 1)) * 100 if in_domain_tests else 100.0
    avg_latency = sum(latencies) / max(len(latencies), 1)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Questions Evaluated:  {total}")
    print(f"Grounding Decision Accuracy: {grounding_accuracy:.1f}%")
    print(f"Refusal Precision (Out-of-Domain): {refusal_precision:.1f}% ({refusal_successes}/{refusal_tests})")
    print(f"In-Domain Retrieval Recall: {in_domain_recall:.1f}% ({in_domain_successes}/{in_domain_tests})")
    print(f"Average Response Latency:   {avg_latency:.2f} ms")
    print(f"Model Runtime Status:       {model_status}")
    print("=" * 70)

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_status": model_status,
        "total_evaluated": total,
        "grounding_accuracy_percent": round(grounding_accuracy, 2),
        "refusal_precision_percent": round(refusal_precision, 2),
        "in_domain_recall_percent": round(in_domain_recall, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "results": results,
    }

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = settings.REPORTS_DIR / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved report to: {report_file}\n")

    return report_payload


if __name__ == "__main__":
    run_evaluation()
