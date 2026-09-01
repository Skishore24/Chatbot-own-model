"""
backend/predict.py
----------------------------------------------------
Interactive Command-Line Predictor for Genkit AI V6.
"""

import sys
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.security import security_service
from app.rag.pipeline import get_rag_pipeline
from app.llm.inference import load_model_and_tokenizer
from app.llm.generation import GenerationEngine


def main():
    print("\n" + "=" * 60)
    print(f"  {settings.APP_NAME} (v{settings.APP_VERSION}) — Interactive CLI")
    print("  Ask anything about Genkit (services, tech stack, pricing, team)")
    print("  Type /exit to quit")
    print("=" * 60 + "\n")

    rag = get_rag_pipeline()
    model, tokenizer, config, status = load_model_and_tokenizer()
    engine = GenerationEngine(model, tokenizer)

    session_id = str(uuid.uuid4())[:8]

    while True:
        try:
            query = input(f"[{session_id}] You > ").strip()
            if not query:
                continue
            if query.lower() in ("/quit", "/exit", "exit", "quit"):
                print("Goodbye!")
                break

            # 1. Sanitize & security
            clean_query, is_safe = security_service.sanitize_input(query)
            if not is_safe or security_service.scan_prompt_injection(clean_query):
                print("Genkit AI > Security alert: Input rejected.")
                continue

            # 2. RAG & Grounding
            chunks, confidence, is_grounded = rag.retrieve(clean_query)

            print(f"Genkit AI [Confidence: {confidence:.2f}] > ", end="", flush=True)

            if not is_grounded:
                print(rag.get_refusal_answer())
            elif status == "MODEL_READY" and engine.model is not None:
                prompt = rag.build_prompt(clean_query, chunks)
                try:
                    llm_ans = engine.generate(prompt, max_new_tokens=settings.MAX_NEW_TOKENS)
                    if llm_ans and len(llm_ans.strip()) > 5:
                        print(llm_ans.strip())
                    else:
                        print(rag.synthesize_answer(clean_query, chunks))
                except Exception:
                    print(rag.synthesize_answer(clean_query, chunks))
            else:
                answer = rag.synthesize_answer(clean_query, chunks)
                print(answer)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI.")
            break


if __name__ == "__main__":
    main()
