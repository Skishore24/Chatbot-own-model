"""
predict.py
----------------------------------------------------
Genkit AI - Interactive CLI Predictor
Usage
-----
    cd backend
    python predict.py

What this does
--------------
Starts an interactive command-line chat session
against the trained Genkit AI model. Use this to
test the full pipeline (domain guard, intent,
retrieval, generation) before deploying.

Commands
--------
  /quit or /exit  : Exit the predictor
  /clear          : Clear session memory
  /status         : Show pipeline status
  /domain <text>  : Test domain guard directly
  /intent <text>  : Test intent classifier directly
Author : Genkit AI
"""
import os
import sys
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import logger, MODEL_READY


def print_banner():
    print("\n" + "=" * 60)
    print("  GENKIT AI - Interactive Predictor")
    print("  Type /help for commands, /quit to exit")
    print("=" * 60 + "\n")


def print_help():
    print("""
Commands:
  /quit       Exit the predictor
  /clear      Clear conversation memory
  /status     Show pipeline status
  /domain <q> Test domain guard
  /intent <q> Test intent classifier
  /entity <q> Test entity extractor
  /help       Show this help
""")


def run_command(cmd: str, args: str, session_id: str):
    """Handle slash-commands."""
    if cmd in ("/quit", "/exit"):
        print("Goodbye!")
        sys.exit(0)

    elif cmd == "/clear":
        from chatbot import clear_memory
        clear_memory(session_id)
        print("[Memory cleared]")

    elif cmd == "/status":
        from chatbot import chatbot_health
        status = chatbot_health()
        print("\nPipeline Status:")
        for k, v in status.items():
            print(f"  {k:<25}: {v}")
        print()

    elif cmd == "/domain":
        from ai.nlp.domain_guard import domain_guard
        result = domain_guard.classify(args)
        print(f"  in_domain  : {result['in_domain']}")
        print(f"  confidence : {result['confidence']:.3f}")
        print(f"  positive   : {result['positive_hits']}")
        print(f"  negative   : {result['negative_hits']}")

    elif cmd == "/intent":
        from ai.nlp.intent_classifier import intent_classifier
        result = intent_classifier.classify(args)
        print(f"  intent     : {result['intent']}")
        print(f"  confidence : {result['confidence']:.3f}")
        print(f"  entities   : {result['entities']}")

    elif cmd == "/entity":
        from ai.nlp.entity_extractor import entity_extractor
        result = entity_extractor.extract(args)
        for k, v in result.items():
            if v:
                print(f"  {k}: {v}")

    elif cmd == "/help":
        print_help()

    else:
        print(f"Unknown command: {cmd}")


def main():
    if not MODEL_READY:
        print("\n⚠️  Model not found. Please run: python train.py")
        print("   (The predictor will still run but responses will be poor)\n")

    print_banner()

    # Initialize pipeline
    print("Loading pipeline...")
    try:
        from chatbot import initialize_chatbot, chatbot_health
        initialize_chatbot()
        health = chatbot_health()
        print(f"✓ Pipeline ready | components: {sum(1 for v in health.values() if v)}/{len(health)}")
    except Exception as e:
        print(f"✗ Pipeline initialization error: {e}")
        print("  Continuing anyway...\n")

    session_id = str(uuid.uuid4())
    print(f"Session: {session_id[:8]}...\n")

    from chatbot import get_answer

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            run_command(cmd, args, session_id)
            continue

        # Normal chat
        print("AI: ", end="", flush=True)
        try:
            for chunk in get_answer(user_input, session_id):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n[Error: {e}]")
        print()


if __name__ == "__main__":
    main()
