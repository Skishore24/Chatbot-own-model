"""
backend/training/evaluate.py
----------------------------------------------------
Benchmark evaluation suite for Genkit AI V6 (Model + Hybrid RAG).
Evaluates retrieval accuracy, groundedness, out-of-domain refusal, and latency.
Generates evaluation_report.json and evaluation_report.md.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger
from app.rag.pipeline import HybridRAGPipeline, get_rag_pipeline
from app.rag.grounding import DOMAIN_REFUSAL_MESSAGE

# Benchmark evaluation dataset with 50+ diverse test cases across all categories
BENCHMARK_QUESTIONS = [
    # 1. Company & Founders
    {"q": "Who founded Genkit?", "cat": "Company", "expected_kw": ["kishore", "hari", "founder", "genkit"]},
    {"q": "What is Genkit?", "cat": "Company", "expected_kw": ["company", "software", "agency", "technology", "solutions"]},
    {"q": "Where is Genkit located?", "cat": "Company", "expected_kw": ["location", "india", "remote", "contact"]},
    {"q": "What is Genkit's mission and vision?", "cat": "Company", "expected_kw": ["mission", "vision", "solutions", "empower"]},
    {"q": "Tell me about the founders of Genkit.", "cat": "Company", "expected_kw": ["kishore", "hari", "founder"]},
    
    # 2. Services
    {"q": "What services does Genkit provide?", "cat": "Services", "expected_kw": ["web", "graphic", "design", "development", "seo", "video"]},
    {"q": "Does Genkit develop mobile apps?", "cat": "Services", "expected_kw": ["mobile", "app", "development", "react"]},
    {"q": "Can Genkit build custom AI models?", "cat": "Services", "expected_kw": ["ai", "chatbot", "custom", "automation"]},
    {"q": "Do you offer UI/UX design services?", "cat": "Services", "expected_kw": ["ui", "ux", "design", "figma"]},
    {"q": "Does Genkit provide video editing and graphic design?", "cat": "Services", "expected_kw": ["video", "graphic", "design", "editing"]},
    {"q": "Can you help with digital marketing and SEO?", "cat": "Services", "expected_kw": ["marketing", "seo", "digital", "rankings"]},
    {"q": "Do you build full-stack SaaS web applications?", "cat": "Services", "expected_kw": ["web", "development", "react", "fastapi"]},

    # 3. Technologies
    {"q": "What tech stack does Genkit use?", "cat": "Technologies", "expected_kw": ["react", "python", "fastapi", "django", "node", "figma"]},
    {"q": "Do you use React for frontend?", "cat": "Technologies", "expected_kw": ["react", "frontend", "interface"]},
    {"q": "What backend frameworks do you specialize in?", "cat": "Technologies", "expected_kw": ["python", "fastapi", "django", "node"]},
    {"q": "What creative design tools do you use?", "cat": "Technologies", "expected_kw": ["figma", "photoshop", "illustrator", "premiere"]},
    {"q": "What databases does Genkit work with?", "cat": "Technologies", "expected_kw": ["mysql", "mongodb", "postgresql", "database"]},

    # 4. Pricing & Process
    {"q": "How much does a website cost at Genkit?", "cat": "Pricing", "expected_kw": ["price", "pricing", "cost", "website"]},
    {"q": "What are your mobile app development rates?", "cat": "Pricing", "expected_kw": ["pricing", "cost", "mobile", "app"]},
    {"q": "What is Genkit's project development process?", "cat": "Process", "expected_kw": ["process", "discovery", "development", "testing"]},
    {"q": "How long does it take to build a custom MVP?", "cat": "Process", "expected_kw": ["timeline", "weeks", "development"]},
    {"q": "Do you offer free consultation for new projects?", "cat": "Pricing", "expected_kw": ["consultation", "free", "quote"]},

    # 5. Portfolio & Projects
    {"q": "What projects has Genkit delivered?", "cat": "Portfolio", "expected_kw": ["project", "portfolio", "client", "platform"]},
    {"q": "Can you show me case studies or past work?", "cat": "Portfolio", "expected_kw": ["case study", "portfolio", "projects"]},
    {"q": "Have you built enterprise AI chatbots before?", "cat": "Portfolio", "expected_kw": ["chatbot", "ai", "model"]},

    # 6. Contact & Support
    {"q": "How can I contact Genkit?", "cat": "Contact", "expected_kw": ["email", "contact", "support", "reach"]},
    {"q": "What is the official email address of Genkit?", "cat": "Contact", "expected_kw": ["email", "contact@genkit.in", "genkit"]},
    {"q": "How do I get a quote for my project?", "cat": "Contact", "expected_kw": ["quote", "contact", "details"]},

    # 7. Out-of-Domain (Must be rejected)
    {"q": "What is the capital of France?", "cat": "OutOfDomain", "is_ood": True},
    {"q": "Who won the football World Cup in 2022?", "cat": "OutOfDomain", "is_ood": True},
    {"q": "What is the recipe for chocolate chip cookies?", "cat": "OutOfDomain", "is_ood": True},
    {"q": "Explain quantum mechanics and Schrödinger's equation.", "cat": "OutOfDomain", "is_ood": True},
    {"q": "What is the stock price of Tesla today?", "cat": "OutOfDomain", "is_ood": True},
]


def run_benchmark() -> dict:
    """Executes benchmark evaluation on the Hybrid RAG pipeline."""
    pipeline = get_rag_pipeline()
    logger.info("=" * 70)
    logger.info(f"RUNNING GENKIT AI V6 BENCHMARK EVALUATION ({len(BENCHMARK_QUESTIONS)} Questions)")
    logger.info("=" * 70)

    total = len(BENCHMARK_QUESTIONS)
    in_domain_total = 0
    in_domain_hits = 0
    ood_total = 0
    ood_correct_refusals = 0
    latencies = []
    category_scores: Dict[str, List[float]] = {}
    detailed_results = []

    for item in BENCHMARK_QUESTIONS:
        query = item["q"]
        category = item["cat"]
        is_ood = item.get("is_ood", False)

        start_time = time.time()
        chunks, confidence, is_grounded = pipeline.retrieve(query, top_k=3)
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)

        if is_ood:
            ood_total += 1
            # Refusal is correct if not grounded
            correct = not is_grounded
            if correct:
                ood_correct_refusals += 1
            score = 1.0 if correct else 0.0
        else:
            in_domain_total += 1
            expected_kw = item.get("expected_kw", [])
            retrieved_text = " ".join([f"{c.title} {c.text}" for c in chunks]).lower()
            hit = any(kw.lower() in retrieved_text for kw in expected_kw) if expected_kw else (len(chunks) > 0)
            if hit:
                in_domain_hits += 1
            score = 1.0 if hit and is_grounded else 0.0

        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)

        detailed_results.append({
            "question": query,
            "category": category,
            "is_out_of_domain": is_ood,
            "grounded": is_grounded,
            "confidence": confidence,
            "retrieved_chunks": [c.title for c in chunks],
            "latency_ms": round(latency_ms, 2),
            "passed": bool(score == 1.0),
        })

    retrieval_acc = (in_domain_hits / max(in_domain_total, 1)) * 100.0
    refusal_acc = (ood_correct_refusals / max(ood_total, 1)) * 100.0
    avg_latency = sum(latencies) / max(len(latencies), 1)

    category_summary = {
        cat: f"{(sum(scores)/len(scores))*100:.1f}% ({len(scores)} tests)"
        for cat, scores in category_scores.items()
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": total,
        "in_domain_retrieval_accuracy": f"{retrieval_acc:.1f}%",
        "out_of_domain_refusal_accuracy": f"{refusal_acc:.1f}%",
        "average_retrieval_latency_ms": f"{avg_latency:.2f} ms",
        "categories": category_summary,
        "results": detailed_results,
    }

    # Save JSON report
    report_json_path = settings.BASE_DIR / "evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    report_md_path = settings.BASE_DIR / "evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Genkit AI v6.0 Evaluation Benchmark Report\n\n")
        f.write(f"**Date:** {report['timestamp']}\n\n")
        f.write(f"- **Total Questions Evaluated:** {total}\n")
        f.write(f"- **In-Domain Retrieval Accuracy:** {report['in_domain_retrieval_accuracy']}\n")
        f.write(f"- **Out-of-Domain Refusal Accuracy:** {report['out_of_domain_refusal_accuracy']}\n")
        f.write(f"- **Average Retrieval Latency:** {report['average_retrieval_latency_ms']}\n\n")
        f.write("### Category Breakdown\n\n")
        for cat, acc in category_summary.items():
            f.write(f"- **{cat}:** {acc}\n")

    logger.info(f"Evaluation report saved to {report_json_path} and {report_md_path}")
    print("\n" + "=" * 60)
    print("  GENKIT AI V6 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  In-Domain Retrieval Accuracy   : {retrieval_acc:.1f}%")
    print(f"  Out-of-Domain Refusal Accuracy : {refusal_acc:.1f}%")
    print(f"  Average Retrieval Latency      : {avg_latency:.2f} ms")
    print("-" * 60)
    for cat, acc in category_summary.items():
        print(f"  - {cat:<20}: {acc}")
    print("=" * 60 + "\n")

    return report


if __name__ == "__main__":
    run_benchmark()
