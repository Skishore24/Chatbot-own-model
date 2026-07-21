"""
ai/nlp/entity_extractor.py
----------------------------------------------------
Genkit AI - Entity Extractor (v4.0)

Extracts domain-specific entities (services, technologies, prices)
and maps them to parent layers (e.g., FastAPI -> Python/Backend).

Author : Genkit AI
"""

import re
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger("Genkit AI")

# Entity Maps for slot filling
SYNONYM_MAP = {
    # Services
    "website development": "Website Development",
    "web development": "Website Development",
    "webdev": "Website Development",
    "website": "Website Development",
    "web app": "Website Development",
    "landing page": "Website Development",
    "e-commerce": "Website Development",
    "online store": "Website Development",
    "online shop": "Website Development",
    "graphic design": "Graphic Design",
    "logo design": "Branding & Visual Identity",
    "logo": "Branding & Visual Identity",
    "branding": "Branding & Visual Identity",
    "brand identity": "Branding & Visual Identity",
    "video editing": "Video Editing",
    "youtube video": "Video Editing",
    "reel": "Video Editing",
    "shorts": "Video Editing",
    "motion graphics": "Video Editing",
    "seo": "Search Engine Optimization (SEO)",
    "search engine": "Search Engine Optimization (SEO)",
    "google ranking": "Search Engine Optimization (SEO)",
    "ai chatbot": "AI & Chatbot Development",
    "chatbot": "AI & Chatbot Development",
    "ai agent": "AI & Chatbot Development",
}

TECH_MAP = {
    "python": ("Python", "Backend"),
    "fastapi": ("FastAPI", "Backend"),
    "django": ("Django", "Backend"),
    "node": ("Node.js", "Backend"),
    "node.js": ("Node.js", "Backend"),
    "java": ("Java", "Backend"),
    "react": ("React", "Frontend"),
    "react.js": ("React", "Frontend"),
    "mysql": ("MySQL", "Database"),
    "postgres": ("PostgreSQL", "Database"),
    "postgresql": ("PostgreSQL", "Database"),
    "mongodb": ("MongoDB", "Database"),
    "figma": ("Figma", "Design Tool"),
    "photoshop": ("Photoshop", "Design Tool"),
    "illustrator": ("Illustrator", "Design Tool"),
    "premiere": ("Premiere Pro", "Video Tool"),
    "after effects": ("After Effects", "Video Tool"),
}

# Regex Signals
_PRICING_SIGNAL_PATTERN = re.compile(
    r"\b(how\s+much|price|pricing|cost|budget|quotation|quote|estimate|fee|rate|afford|discount|package|charge|invoice|payment)\b",
    re.I,
)
_ACTION_SIGNAL_PATTERN = re.compile(
    r"\b(hire|contact|reach|start\s+a\s+project|get\s+started|work\s+with|partner|collaborate)\b",
    re.I,
)
_PRONOUN_PATTERN = re.compile(
    r"\b(you|your|yours|it|its|they|them|their|theirs|we|our)\b",
    re.I,
)
_EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)

class EntityExtractor:
    """
    NLP Entity Extractor & Slot Filler.
    """

    def __init__(self) -> None:
        self.synonym_map = SYNONYM_MAP
        self.tech_map = TECH_MAP

    def extract(self, query: str) -> Dict:
        """
        Extract services, tech, categories, and other signals from the query.
        """
        normalized = query.lower()
        
        extracted_services = []
        extracted_techs = []
        extracted_categories = []
        
        # 1. Extract Services
        for kw, service_name in self.synonym_map.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', normalized):
                extracted_services.append(service_name)
                
        # 2. Extract Tech & Parent Categories
        for kw, (tech_name, category) in self.tech_map.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', normalized):
                extracted_techs.append(tech_name)
                extracted_categories.append(category)
                
        # Deduplicate
        services = list(dict.fromkeys(extracted_services))
        technologies = list(dict.fromkeys(extracted_techs))
        categories = list(dict.fromkeys(extracted_categories))
        
        # Signals
        pricing_signal = bool(_PRICING_SIGNAL_PATTERN.search(query))
        action_signal = bool(_ACTION_SIGNAL_PATTERN.search(query))
        has_pronouns = bool(_PRONOUN_PATTERN.search(query))
        email_match = _EMAIL_PATTERN.search(query)
        email_in_query = email_match.group() if email_match else None
        
        # Query enrichment for TF-IDF / BM25
        enrichment = []
        if services:
            enrichment.extend(services)
        if technologies:
            enrichment.extend(technologies)
            enrichment.extend(categories)
        if pricing_signal:
            enrichment.append("pricing cost packages budget")
            
        enriched_query = query
        if enrichment:
            enriched_query = query + " " + " ".join(list(dict.fromkeys(enrichment)))
            
        return {
            "services": services,
            "technologies": technologies,
            "categories": categories,
            "pricing_signal": pricing_signal,
            "action_signal": action_signal,
            "has_pronouns": has_pronouns,
            "email_in_query": email_in_query,
            "enriched_query": enriched_query,
        }

    def resolve_pronouns(self, query: str, context_entity: str = "Genkit") -> str:
        """Replace pronouns for query rewriting."""
        resolved = re.sub(r"\byour\b", f"{context_entity}'s", query, flags=re.I)
        resolved = re.sub(r"\byou\b", context_entity, resolved, flags=re.I)
        return resolved

# Singleton
entity_extractor = EntityExtractor()

__all__ = ["EntityExtractor", "entity_extractor"]
