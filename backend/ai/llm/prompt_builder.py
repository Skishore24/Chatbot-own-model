"""
ai/llm/prompt_builder.py
----------------------------------------------------
Genkit AI - Contextual Prompt Builder (v4.0)

Constructs structured prompts incorporating intent, entities,
context documents, and history.

Author : Genkit AI
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger("Genkit AI")

SYSTEM_INSTRUCTION = (
    "You are Genkit AI, the official AI assistant for Genkit digital solutions. "
    "Answer questions accurately using the facts provided in the Context. "
    "Ensure your output uses clear, complete, and grammatically correct sentences that are easy for the user to understand. "
    "If the context is insufficient, decline politely."
)

class PromptBuilder:
    """
    Builds structured prompts for contextual RAG reasoning.
    """

    def __init__(
        self,
        max_context_chars: int = 1000,
        max_history_turns: int = 2,
        max_prompt_chars: int = 2000,
    ) -> None:
        self.max_context_chars = max_context_chars
        self.max_history_turns = max_history_turns
        self.max_prompt_chars = max_prompt_chars

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def _format_context(self, documents: List[Dict]) -> str:
        if not documents:
            return "No context available."
        parts = []
        for doc in documents:
            a = doc.get("answer") or doc.get("text") or ""
            if a:
                parts.append(f"Fact: {a.strip()}")
        return "\n\n".join(parts)

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "No previous history."
        recent = history[-(self.max_history_turns * 2):]
        lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def build(
        self,
        query: str,
        documents: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
        intent: Optional[str] = "general",
        entities: Optional[Dict] = None,
    ) -> str:
        """
        Builds the structured prompt matching the model's training format.
        """
        documents = documents or []
        history = history or []
        entities = entities or {}

        context_text = self._format_context(documents)
        history_text = self._format_history(history)
        
        # Format entities list
        ents_list = []
        for k, v in entities.items():
            if v:
                ents_list.append(f"{k}: {v}")
        ents_str = ", ".join(ents_list) if ents_list else "None"

        # Build prompt
        prompt = (
            f"[INST]\n"
            f"System: {SYSTEM_INSTRUCTION}\n"
            f"Intent: {intent}\n"
            f"Entities: {ents_str}\n\n"
            f"Context:\n{context_text}\n\n"
            f"History:\n{history_text}\n\n"
            f"Question:\n{query.strip()}\n"
            f"[/INST]"
        )

        return prompt

    def build_minimal(self, query: str) -> str:
        return (
            f"[INST]\n"
            f"System: {SYSTEM_INSTRUCTION}\n"
            f"Intent: general\n"
            f"Entities: None\n\n"
            f"Context:\nNo context available.\n\n"
            f"History:\nNo previous history.\n\n"
            f"Question:\n{query.strip()}\n"
            f"[/INST]"
        )

# Singleton
prompt_builder = PromptBuilder()

__all__ = ["PromptBuilder", "prompt_builder", "SYSTEM_INSTRUCTION"]
