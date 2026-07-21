"""
ai/rag/knowledge_graph.py
----------------------------------------------------
Genkit AI - Knowledge Graph Relationship Store

Stores relational links between services, tech stacks, and projects,
allowing the RAG engine to expand context automatically.

Author : Genkit AI
"""

from typing import Dict, List, Set

RELATIONSHIPS = {
    "genkit ai": [
        "website development", "graphic design", "branding & visual identity",
        "video editing", "search engine optimization (seo)", "ai & chatbot development"
    ],
    "website development": [
        "python", "node.js", "java", "react", "fastapi", "django", "express",
        "seo", "hosting", "deployment", "shopsmart"
    ],
    "ai & chatbot development": [
        "python", "fastapi", "pytorch", "mysql", "scikit-learn", "lexibot"
    ],
    "python": ["fastapi", "django", "pytorch", "scikit-learn"],
    "fastapi": ["mysql", "postgresql", "rest apis"],
    "django": ["postgresql", "mysql", "admin portals"],
    "react": ["html5 & css3", "javascript", "shopsmart"],
    "branding & visual identity": ["figma", "adobe illustrator", "logos", "style guides", "zenbites"],
    "graphic design": ["adobe photoshop", "adobe illustrator", "figma", "flyers", "banners"],
    "video editing": ["adobe premiere pro", "adobe after effects", "youtube", "instagram reels", "techstream"],
    "seo": ["google ranking", "search engine optimization", "google analytics", "page speed"]
}

class KnowledgeGraph:
    """
    In-memory knowledge graph representation of the Genkit domain.
    """

    def __init__(self) -> None:
        self.graph = RELATIONSHIPS

    def get_related(self, concept: str) -> List[str]:
        """Get adjacent nodes for a concept (case-insensitive search)."""
        key = concept.lower().strip()
        
        # Exact match
        if key in self.graph:
            return self.graph[key]
            
        # Substring match fallback
        for k, v in self.graph.items():
            if k in key or key in k:
                return v
                
        return []

    def expand_context(self, entities: List[str], max_depth: int = 1) -> List[str]:
        """
        Expand list of input entities to include adjacent related concepts.
        """
        expanded = set()
        for entity in entities:
            expanded.add(entity)
            related = self.get_related(entity)
            for r in related:
                expanded.add(r)
                
        # Format elements cleanly (title case)
        return sorted([e.title() for e in expanded])

# Singleton
knowledge_graph = KnowledgeGraph()

__all__ = ["KnowledgeGraph", "knowledge_graph"]
