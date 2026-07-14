"""
backend/ai/agents/router.py
----------------------------------------------------
Genkit AI - Intelligent Agent Router

Routes user queries to the appropriate specialist.

Agents:
    • FAQ Agent
    • Sales Agent
    • Support Agent

Author : Genkit AI
"""

from config import logger

from ai.agents.faq_agent import FAQAgent
from ai.agents.sales_agent import SalesAgent
from ai.agents.support_agent import SupportAgent


class AgentRouter:
    """
    Intelligent Agent Router

    Detects the user's intent and forwards the query
    to the most suitable specialist agent.
    """

    def __init__(self):

        self.faq_agent = FAQAgent()
        self.sales_agent = SalesAgent()
        self.support_agent = SupportAgent()

        self.default_agent = self.faq_agent

    # ---------------------------------------------------------

    def detect_intent(self, query: str) -> str:
        """
        Detect which agent should answer.

        Returns:
            sales
            support
            faq
        """

        if not query:
            return "faq"

        try:

            if self.sales_agent.can_answer(query):
                return "sales"

            if self.support_agent.can_answer(query):
                return "support"

            if self.faq_agent.can_answer(query):
                return "faq"

        except Exception as e:

            logger.exception(f"Intent Detection Error : {e}")

        return "faq"

    # ---------------------------------------------------------

    def get_agent(self, intent: str):
        """
        Return the correct agent instance.
        """

        if intent == "sales":
            return self.sales_agent

        if intent == "support":
            return self.support_agent

        return self.default_agent

    # ---------------------------------------------------------

    def route(self, query: str) -> str:
        """
        Main routing function.

        Flow:

            User Query
                 │
                 ▼
          Detect Intent
                 │
                 ▼
          Select Agent
                 │
                 ▼
          Generate Response
        """

        try:

            intent = self.detect_intent(query)

            logger.info(f"[Router] Intent Detected : {intent}")

            agent = self.get_agent(intent)

            response = agent.answer(query)

            if not response:

                return (
                    "I'm sorry, I couldn't find an appropriate "
                    "answer for your question."
                )

            return response

        except Exception as e:

            logger.exception(f"Router Error : {e}")

            return (
                "Sorry, an internal routing error occurred. "
                "Please try again."
            )
