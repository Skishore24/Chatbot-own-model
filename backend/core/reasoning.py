from typing import List


def build_reasoning(query: str, context: str) -> str:
    """
    Lightweight reasoning layer (no heavy LLM logic).
    Helps structure thinking before answering.
    """

    steps: List[str] = []

    steps.append(f"User asked: {query}")

    if context:
        steps.append("Relevant company info found")
    else:
        steps.append("No strong context found")

    steps.append("Generate concise answer using context only")

    return " → ".join(steps)