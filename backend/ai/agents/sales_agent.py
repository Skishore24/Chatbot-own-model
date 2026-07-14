"""
backend/ai/agents/sales_agent.py
----------------------------------------------------
Genkit AI - Sales Agent

Handles:
    • Pricing
    • Quotations
    • Business Enquiries
    • Service Requests
    • Lead Detection

Author : Genkit AI
"""

import re
from typing import Optional

from config import logger
from ai.rag.retriever import Retriever
from ai.rag.reranker import Reranker


class SalesAgent:
    """
    Sales Agent

    Handles all business-related queries.
    """

    SALES_KEYWORDS = {
        "price",
        "pricing",
        "cost",
        "quote",
        "quotation",
        "budget",
        "hire",
        "project",
        "website",
        "web development",
        "software",
        "application",
        "mobile app",
        "android",
        "ios",
        "ai",
        "artificial intelligence",
        "machine learning",
        "chatbot",
        "automation",
        "erp",
        "crm",
        "dashboard",
        "portal",
        "development",
        "startup",
        "company"
    }

    HOT_LEAD_KEYWORDS = {
        "hire",
        "quotation",
        "quote",
        "pricing",
        "cost",
        "budget",
        "project",
        "contact me",
        "call me",
        "need website",
        "need app",
        "need chatbot",
        "need ai",
        "build website",
        "build app",
        "software development",
        "web development"
    }

    def __init__(self):

        self.retriever = Retriever()
        self.reranker = Reranker()

    # ---------------------------------------------------------

    def can_answer(self, query: str) -> bool:

        if not query:
            return False

        query = query.lower()

        return any(keyword in query for keyword in self.SALES_KEYWORDS)

    # ---------------------------------------------------------

    def is_hot_lead(self, query: str) -> bool:

        query = query.lower()

        return any(keyword in query for keyword in self.HOT_LEAD_KEYWORDS)

    # ---------------------------------------------------------

    def extract_budget(self, text: str) -> Optional[int]:
        """
        Supports

        ₹50000
        50000
        50k
        2 lakh
        """

        text = text.lower()

        match = re.search(r"(\d+(?:\.\d+)?)\s*(k|lakh|lakhs)?", text)

        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2)

        if unit == "k":
            value *= 1000

        elif unit in ("lakh", "lakhs"):
            value *= 100000

        return int(value)

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
                    "Genkit provides:\n\n"
                    "• Web Development\n"
                    "• Mobile App Development\n"
                    "• AI Solutions\n"
                    "• Chatbot Development\n"
                    "• UI/UX Design\n"
                    "• Graphic Design\n"
                    "• Video Editing\n"
                    "• Business Automation\n"
                )

            budget = self.extract_budget(query)

            if budget:

                response += (
                    f"\n\nEstimated Budget Mentioned: ₹{budget:,}"
                )

            if self.is_hot_lead(query):

                response += (
                    "\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "📩 Business Enquiry\n\n"
                    "Please share the following details:\n\n"
                    "• Name\n"
                    "• Company\n"
                    "• Email\n"
                    "• Phone Number\n"
                    "• Project Description\n"
                    "• Budget\n"
                    "• Expected Delivery Date\n\n"
                    "Our business team will contact you shortly."
                )

            return response

        except Exception as e:

            logger.exception(f"Sales Agent Error : {e}")

            return (
                "Sorry, something went wrong while processing "
                "your business enquiry."
            )
