"""
backend/predict.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Interactive CLI Predictor
"""

import sys
import uuid
from pathlib import Path

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger
from app.core.security import security_service
from app.ai.llm.model_loader import get_model_and_tokenizer
from app.ai.llm.inference import GenerationEngine
from app.ai.llm.prompt_builder import prompt_builder
from app.ai.rag.retriever import HybridRetriever

# Initialize Engine
gpt_model, tokenizer, gpt_config = get_model_and_tokenizer()
gen_engine = GenerationEngine(gpt_model, tokenizer)
retriever = HybridRetriever()


def print_banner():
    print("\n" + "=" * 60)
    print(f"  {settings.APP_NAME} (v{settings.APP_VERSION}) — Interactive CLI Predictor")
    print("  Type /quit to exit")
    print("=" * 60 + "\n")


def main():
    print_banner()
    session_id = str(uuid.uuid4())[:8]

    while True:
        try:
            query = input(f"\n[{session_id}] User > ").strip()
            if not query:
                continue

            if query.lower() in ("/quit", "/exit"):
                print("Goodbye!")
                break

            # Security Scan
            cleaned, is_safe = security_service.sanitize_input(query)
            if not is_safe or security_service.scan_prompt_injection(cleaned):
                print("Genkit AI > Security alert: Input rejected due to security policy.")
                continue

            # Retrieval & Generation
            context_blocks, _ = retriever.retrieve(cleaned)
            prompt = prompt_builder.build_prompt(cleaned, context_passages=context_blocks)

            print("Genkit AI > ", end="", flush=True)
            for chunk in gen_engine.generate_stream(prompt):
                print(chunk, end="", flush=True)
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting predictor.")
            break


if __name__ == "__main__":
    main()
