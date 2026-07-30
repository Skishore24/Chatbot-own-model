"""
backend/app/ai/llm/prompt_builder.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Structured Prompt Builder
Formats RAG retrieved context, history turns, and query tags using special control tokens.
"""

from typing import Dict, List, Optional


class PromptBuilder:
    """Enterprise Prompt Compiler for Genkit AI v5.0."""

    SYSTEM_INSTRUCTION = (
        "You are Genkit AI, an enterprise AI assistant for Genkit.in (AI, custom software, web, and mobile app development). "
        "Your task is to answer user queries strictly grounded in the retrieved enterprise context. "
        "Do not answer general out-of-scope questions (weather, sports, politics, non-Genkit coding). "
        "Be helpful, precise, professional, and friendly."
    )

    def build_prompt(
        self,
        query: str,
        context_passages: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Compiles input into structured control-token tagged prompt string.
        """
        sys_inst = system_instruction or self.SYSTEM_INSTRUCTION
        prompt_parts: List[str] = []

        # System Header
        prompt_parts.append(f"System: {sys_inst}\n")

        # 1. RAG Context Block
        if context_passages:
            prompt_parts.append("<context_start>")
            for idx, passage in enumerate(context_passages, 1):
                prompt_parts.append(f"[{idx}] {passage.strip()}")
            prompt_parts.append("<context_end>\n")

        # 2. Conversation History Block
        if history:
            prompt_parts.append("<history_start>")
            for turn in history[-4:]:  # Keep last 4 turns max
                role = turn.get("role", "user").capitalize()
                text = turn.get("text") or turn.get("content") or ""
                prompt_parts.append(f"{role}: {text}")
            prompt_parts.append("<history_end>\n")

        # 3. User Query Block
        prompt_parts.append("<query_start>")
        prompt_parts.append(f"User Question: {query.strip()}")
        prompt_parts.append("<query_end>\n")

        # 4. Assistant Answer Target Trigger
        prompt_parts.append("<ans_start>\nGenkit AI:")

        return "\n".join(prompt_parts)


prompt_builder = PromptBuilder()
