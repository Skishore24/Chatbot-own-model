"""
backend/app/ai/rag/knowledge_graph.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise GraphRAG Sub-Graph Extraction Engine
BFS Entity Relation Traversal on Adjacency Matrix for Multi-Hop Context Assembly.
"""

from collections import deque
from typing import Dict, List, Set, Tuple

from app.core.logger import logger


class KnowledgeGraphEngine:
    """Enterprise GraphRAG Entity-Relation Search Engine."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, str]] = {}
        self.adj_list: Dict[str, List[Tuple[str, str]]] = {}
        self._build_default_graph()

    def _build_default_graph(self):
        """Populates company domain knowledge entity nodes and relationships."""
        entities = [
            ("Genkit", "Company", "Genkit.in is an AI and custom software development company founded in 2024."),
            ("AI_Development", "Service", "Custom LLM engineering, PyTorch models, RAG systems, and AI automation."),
            ("Web_Development", "Service", "Enterprise React, Next.js, FastAPI, Node.js, and cloud backends."),
            ("Mobile_Development", "Service", "Cross-platform Flutter and React Native mobile apps."),
            ("PyTorch", "Technology", "Primary deep learning framework used for custom GPT and neural models."),
            ("React", "Technology", "Frontend web framework used for interactive AI widget dashboards."),
            ("FastAPI", "Technology", "High-performance Python async REST & streaming backend API framework."),
            ("MySQL", "Technology", "Relational database used for session history, leads, and persistent user data."),
            ("Starter_Plan", "Pricing", "Flexible project estimates starting from basic tier minimums."),
            ("Enterprise_Plan", "Pricing", "Dedicated AI architecture design and multi-GPU infrastructure setups."),
        ]

        for entity_id, entity_type, desc in entities:
            self.nodes[entity_id] = {"type": entity_type, "description": desc}
            self.adj_list[entity_id] = []

        relations = [
            ("Genkit", "PROVIDES_SERVICE", "AI_Development"),
            ("Genkit", "PROVIDES_SERVICE", "Web_Development"),
            ("Genkit", "PROVIDES_SERVICE", "Mobile_Development"),
            ("AI_Development", "BUILT_WITH", "PyTorch"),
            ("Web_Development", "BUILT_WITH", "React"),
            ("Web_Development", "BUILT_WITH", "FastAPI"),
            ("Web_Development", "PERSISTS_TO", "MySQL"),
            ("Genkit", "HAS_PRICING", "Starter_Plan"),
            ("Genkit", "HAS_PRICING", "Enterprise_Plan"),
        ]

        for src, rel, dst in relations:
            if src in self.adj_list and dst in self.adj_list:
                self.adj_list[src].append((rel, dst))
                self.adj_list[dst].append((f"REV_{rel}", src))

    def extract_subgraph_facts(self, query_entities: List[str], max_depth: int = 2) -> List[str]:
        """
        Performs BFS graph traversal up to max_depth starting from extracted query entity nodes.
        Returns a list of structured graph fact statements.
        """
        if not query_entities:
            return []

        visited_nodes: Set[str] = set()
        visited_edges: Set[Tuple[str, str, str]] = set()
        facts: List[str] = []

        queue = deque([(entity, 0) for entity in query_entities if entity in self.nodes])

        while queue:
            curr_node, depth = queue.popleft()
            if curr_node not in visited_nodes:
                visited_nodes.add(curr_node)
                node_info = self.nodes.get(curr_node)
                if node_info:
                    facts.append(f"Entity [{curr_node}] ({node_info['type']}): {node_info['description']}")

            if depth < max_depth:
                for rel, neighbor in self.adj_list.get(curr_node, []):
                    edge_key = (curr_node, rel, neighbor)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        rel_clean = rel.replace("REV_", "")
                        facts.append(f"Fact: {curr_node} --[{rel_clean}]--> {neighbor}")

                    if neighbor not in visited_nodes:
                        queue.append((neighbor, depth + 1))

        return facts
