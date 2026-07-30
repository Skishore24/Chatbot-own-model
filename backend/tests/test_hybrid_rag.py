"""
backend/tests/test_hybrid_rag.py
----------------------------------------------------
Unit tests for GraphRAG, CandidateReranker, ContextBuilder, and HybridRetriever.
"""

import sys
import unittest
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.rag.knowledge_graph import KnowledgeGraphEngine
from app.ai.rag.reranker import CandidateReranker
from app.ai.rag.context_builder import context_builder
from app.ai.rag.retriever import HybridRetriever


class TestHybridRAG(unittest.TestCase):

    def setUp(self):
        self.graph = KnowledgeGraphEngine()
        self.reranker = CandidateReranker()
        self.retriever = HybridRetriever()

    def test_knowledge_graph_traversal(self):
        """Verify BFS sub-graph fact extraction."""
        facts = self.graph.extract_subgraph_facts(["Genkit"], max_depth=2)
        self.assertTrue(len(facts) > 0)
        self.assertTrue(any("PROVIDES_SERVICE" in f for f in facts))

    def test_candidate_reranker(self):
        """Verify candidate reranking order."""
        candidates = [
            {"text": "General coding tutorials Python", "score": 0.1},
            {"text": "Genkit AI Development Services PyTorch", "score": 0.5},
        ]
        reranked = self.reranker.rerank_passages("Genkit AI Services", candidates, top_n=2)
        self.assertEqual(len(reranked), 2)
        self.assertIn("Genkit AI Development", reranked[0]["text"])

    def test_hybrid_retriever(self):
        """Verify full hybrid retrieval pipeline."""
        context_blocks, reranked = self.retriever.retrieve("What are Genkit AI services?")
        self.assertIsInstance(context_blocks, list)
        self.assertIsInstance(reranked, list)


if __name__ == "__main__":
    unittest.main()
