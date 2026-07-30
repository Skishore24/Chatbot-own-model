"""
backend/app/ai/rag/context_builder.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Parent-Child Context Compiler
Aggregates top candidate chunks and KnowledgeGraph facts into a clean structured context string.
"""

from typing import Dict, List, Optional


class ContextBuilder:
    """Assembles parent-child document chunks and GraphRAG facts into prompt blocks."""

    def build_context_block(
        self,
        ranked_passages: List[Dict[str, str]],
        graph_facts: Optional[List[str]] = None,
        max_chars: int = 3000,
    ) -> List[str]:
        """
        Compiles ranked passages and sub-graph facts into clean passage strings.
        """
        compiled_blocks: List[str] = []
        total_chars = 0

        # Include GraphRAG Entity Facts
        if graph_facts:
            graph_summary = "Graph Entity Facts: " + " | ".join(graph_facts[:4])
            compiled_blocks.append(graph_summary)
            total_chars += len(graph_summary)

        # Include Top Document Chunks
        for idx, item in enumerate(ranked_passages, 1):
            text = item.get("text", "").strip()
            category = item.get("category", "General")
            block = f"[{category}] {text}"

            if total_chars + len(block) > max_chars:
                break

            compiled_blocks.append(block)
            total_chars += len(block)

        return compiled_blocks


context_builder = ContextBuilder()
