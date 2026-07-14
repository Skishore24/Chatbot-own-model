"""
backend/ai/agents/support_agent.py
----------------------------------------------------
Genkit AI - Support Agent

Handles:
    • Technical Support
    • Login Problems
    • Website Issues
    • Server Errors
    • Database Problems
    • Hosting Issues
    • Bug Reports

Author : Genkit AI
"""

import uuid

from config import logger
from ai.rag.retriever import Retriever
from ai.rag.reranker import Reranker


class SupportAgent:
    """
    Technical Support Agent
    """

    SUPPORT_KEYWORDS = {
        "support",
        "help",
        "issue",
        "problem",
        "error",
        "bug",
        "failed",
        "failure",
        "crash",
        "login",
        "password",
        "account",
        "website",
        "website down",
        "server",
        "hosting",
        "database",
        "mysql",
        "mongodb",
        "payment",
        "technical",
        "not working",
        "unable",
        "cannot",
        "exception"
    }

    HIGH_PRIORITY = {
        "server",
        "database",
        "website down",
        "payment",
        "crash",
        "security",
        "login failed"
    }

    def __init__(self):

        self.retriever = Retriever()
        self.reranker = Reranker()

    # ---------------------------------------------------------

    def can_answer(self, query: str) -> bool:

        if not query:
            return False

        query = query.lower()

        return any(word in query for word in self.SUPPORT_KEYWORDS)

    # ---------------------------------------------------------

    def detect_priority(self, query: str) -> str:

        query = query.lower()

        if any(word in query for word in self.HIGH_PRIORITY):
            return "High"

        return "Normal"

    # ---------------------------------------------------------

    def answer(self, query: str) -> str:

        try:

            docs = self.retriever.retrieve(
                query=query,
                top_k=5
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

            if responses:

                response = "\n\n".join(responses)

            else:

                response = (
                    "I'm sorry that you're experiencing a problem.\n\n"
                    "To help you quickly, please provide:\n\n"
                    "• Problem Description\n"
                    "• Screenshot (if available)\n"
                    "• Error Message\n"
                    "• Browser or Device\n"
                    "• Steps to reproduce the issue\n"
                )

            priority = self.detect_priority(query)

            response += (
                "\n\n"
                f"Support Priority : {priority}"
            )

            return response

        except Exception as e:

            logger.exception(f"Support Agent Error : {e}")

            return (
                "Sorry, an unexpected error occurred while "
                "processing your support request."
            )

    # ---------------------------------------------------------

    def create_ticket(
        self,
        name: str,
        email: str,
        issue: str
    ) -> dict:
        """
        Create a support ticket.
        Later this can be saved into MySQL.
        """

        ticket = {
            "ticket_id": str(uuid.uuid4())[:8].upper(),
            "name": name,
            "email": email,
            "issue": issue,
            "priority": self.detect_priority(issue),
            "status": "Open"
        }

        logger.info(
            f"Support Ticket Created : {ticket['ticket_id']}"
        )

        return ticket
