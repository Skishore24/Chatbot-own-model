"""
backend/ai/agents/faq_agent.py
----------------------------------------------------
Genkit AI - FAQ Agent

Handles:
    • Company Information
    • Services
    • Contact Details
    • Portfolio
    • Technologies
    • General FAQs

Uses:
    Retriever
    Reranker

Author : Genkit AI
"""

from config import logger
from ai.rag.retriever import Retriever
from ai.rag.reranker import Reranker


class FAQAgent:
    """
    FAQ Agent

    Responsible for answering company-related questions
    using the Genkit knowledge base.
    """

    def __init__(self):

        self.retriever = Retriever()
        self.reranker = Reranker()

        self.keywords = {
            "genkit",
            "company",
            "about",
            "service",
            "services",
            "portfolio",
            "website",
            "contact",
            "email",
            "phone",
            "address",
            "location",
            "technology",
            "team",
            "mission",
            "vision",
            "pricing",
            "support",
            "projects",
            "products"
        }

    # ---------------------------------------------------------

    def can_answer(self, query: str) -> bool:
        """
        Determines whether this agent should answer.
        """

        if not query:
            return False

        query = query.lower()

        return any(word in query for word in self.keywords)

    # ---------------------------------------------------------

    def answer(self, query: str) -> str:
        """
        Retrieve and return the best FAQ answer.
        """

        try:

            docs = self.retriever.retrieve(
                query=query,
                top_k=5
            )

            if not docs:
                return (
                    "I couldn't find any information related to your question "
                    "in the Genkit knowledge base."
                )

            docs = self.reranker.rerank(
                query=query,
                documents=docs,
                top_k=3
            )

            responses = []
            seen = set()

            for doc in docs:

                if not isinstance(doc, dict):
                    continue

                text = doc.get("text", "").strip()

                if not text:
                    continue

                if text.lower() in seen:
                    continue

                seen.add(text.lower())

                responses.append(text)

            if not responses:
                return (
                    "Sorry, I couldn't find a suitable answer "
                    "for your question."
                )

            return "\n\n".join(responses)

        except Exception as e:

            logger.exception(
                f"FAQ Agent Error : {e}"
            )

            return (
                "Sorry, an internal error occurred while "
                "processing your request."
            )
