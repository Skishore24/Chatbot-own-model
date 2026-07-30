"""
scripts/evaluate.py
----------------------------------------------------
GENKIT AI v5.0 Model & RAG Evaluation Benchmark CLI Script
Usage:
    python scripts/evaluate.py
"""

import sys
import time
from pathlib import Path

# Add backend directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger
from app.ai.tokenizer.tokenizer import default_tokenizer
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.llm.inference import GenerationEngine
from app.ai.rag.retriever import HybridRetriever

EVAL_QUESTIONS = [
    "What services does Genkit offer?",
    "How much does a website cost at Genkit?",
    "How can I contact Genkit?",
    "Does Genkit make mobile apps?",
    "What technologies does Genkit use?",
]


def main():
    logger.info("=" * 70)
    logger.info(f"{settings.APP_NAME} — MODEL & RAG EVALUATION SUITE")
    logger.info("=" * 70)

    gpt_config = GPTConfig(vocab_size=16000, block_size=2048, n_embd=384, n_head=8, n_kv_head=2, n_layer=8)
    model = EnterpriseGPTModel(gpt_config)
    retriever = HybridRetriever()
    engine = GenerationEngine(model, default_tokenizer)

    logger.info("Running evaluation queries...")
    for idx, question in enumerate(EVAL_QUESTIONS, 1):
        start = time.time()
        context_blocks, reranked = retriever.retrieve(question, top_k=3)
        latency = (time.time() - start) * 1000

        logger.info(f"[{idx}/{len(EVAL_QUESTIONS)}] Question: {question}")
        logger.info(f"      Retrieval Latency: {latency:.2f}ms | Chunks Found: {len(context_blocks)}")

    logger.info("=" * 70)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
