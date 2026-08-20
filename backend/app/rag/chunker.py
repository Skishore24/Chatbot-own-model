"""
backend/app/rag/chunker.py
----------------------------------------------------
Document and Chunk data structures with metadata for Genkit AI V6.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DocumentChunk:
    """Represents a coherent piece of domain knowledge."""

    id: str
    source: str
    category: str
    title: str
    text: str
    keywords: List[str] = field(default_factory=list)
    priority: int = 1
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "category": self.category,
            "title": self.title,
            "text": self.text,
            "keywords": self.keywords,
            "priority": self.priority,
            "score": round(self.score, 4),
        }
