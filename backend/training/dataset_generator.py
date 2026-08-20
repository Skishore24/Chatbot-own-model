"""
backend/training/dataset_generator.py
----------------------------------------------------
Genkit AI V6 - Structured Dataset Generator
Generates high-quality training pairs (instruction, output)
from structured domain knowledge files in backend/datasets/*.json.
"""

import os
import sys
import json
import random
from pathlib import Path

# Setup paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger

DATASET_DIR = settings.DATASET_DIR
SYSTEM_INSTRUCTION = "You are Genkit AI, an enterprise AI assistant for Genkit.in."


def format_training_instruction(query: str, intent: str, entities_str: str, context_text: str) -> str:
    """Formats the instruction block for model training."""
    return (
        f"System: {SYSTEM_INSTRUCTION}\n"
        f"Intent: {intent}\n"
        f"Entities: {entities_str}\n\n"
        f"Context:\n{context_text}\n\n"
        f"History:\nNo previous history.\n\n"
        f"Question:\n{query}"
    )


PREFIXES = [
    "", "tell me ", "can you explain ", "i want to know ", "please describe ",
    "give me info about ", "what is ", "explain ", "how does ", "show me "
]

SUFFIXES = [
    "", " please", " now", " for my business", " details", " options"
]


def load_structured_kb():
    """Loads all structured JSON files in datasets directory."""
    kb = {}
    kb_files = [
        "company.json", "services.json", "projects.json", "technologies.json",
        "pricing.json", "team.json", "clients.json", "portfolio.json",
        "faq.json", "policies.json", "contact.json"
    ]
    for filename in kb_files:
        path = DATASET_DIR / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            kb[filename.split(".")[0]] = json.load(f)
    return kb


def generate_samples(kb):
    """Generates synthetic multi-turn QA instruction samples."""
    samples = []
    
    # 1. greeting
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings", "whats up", "hola"]
    for g in greetings:
        for pref in ["", "hey ", "hello "]:
            for suff in ["", " there", " team", " friend"]:
                query = f"{pref}{g}{suff}".strip()
                ans = (
                    "Reasoning: The user is greeting the assistant. I should respond politely and introduce Genkit AI.\n\n"
                    "Answer:\nHello and welcome to Genkit AI! "
                    "We are a premier digital solutions agency here to help you grow your business. "
                    "Our services cover custom website development, graphic design, branding, SEO, video editing, and AI chatbot automation. "
                    "We stand ready to guide you from project design to launch. "
                    "What digital services can we help you with today?"
                )
                inst = format_training_instruction(query, "greeting", "None", "No context available.")
                samples.append({"instruction": inst, "intent": "greeting", "output": ans})

    # 2. out_of_domain
    out_of_domain_topics = [
        "Elon Musk", "cricket score", "weather", "recipe for pasta", "Bitcoin", "cryptocurrency", 
        "Prime Minister", "FIFA World Cup", "solve math homework", "write a love poem", "how to bake a cake", 
        "stock market today", "best action movies", "popular pop songs", "gravity", "quantum physics",
        "buy real estate", "cat food brands", "booking flights to Paris", "what is the capital of France"
    ]
    for topic in out_of_domain_topics:
        for pref in PREFIXES:
            for suff in SUFFIXES:
                query = f"{pref}{topic}{suff}".strip()
                ans = (
                    "Reasoning: The user query falls outside of Genkit AI's business scope. I must refuse politely and redirect them to our services.\n\n"
                    "Answer:\nI am Genkit AI, designed exclusively to assist with questions about Genkit's digital services, company, and team. "
                    "I am not able to help with this topic. "
                    "We specialize in website development, UI/UX, branding, SEO, video production, and custom AI bots. "
                    "Please feel free to ask me anything about Genkit!"
                )
                inst = format_training_instruction(query, "out_of_domain", "None", "No context available.")
                samples.append({"instruction": inst, "intent": "out_of_domain", "output": ans})

    # 3. contact
    if "contact" in kb:
        contact_templates = [
            "how can i contact you", "what is your email", "contact channels", "reach out to genkit",
            "how quickly do you respond", "book a consultation", "where are you located", "do you have instagram"
        ]
        contact_data = kb["contact"]
        channels_str = ", ".join([c["type"] for c in contact_data.get("channels", [])])
        context_text = f"Primary email: {contact_data.get('email', 'contact@genkit.in')}. Website: {contact_data.get('website', 'genkit.in')}. Channels: {channels_str}."
        for temp in contact_templates:
            for p in PREFIXES:
                for s in SUFFIXES:
                    query = f"{p}{temp}{s}".strip()
                    ans = (
                        "Reasoning: The user is asking how to contact Genkit AI. I need to supply our primary email, social channels, and consultation details.\n\n"
                        "Answer:\nYou can contact Genkit AI easily via email at contact@genkit.in.\n\n"
                        "Our team works from India, serving clients across the globe. You can also find us on Instagram or check our projects on GitHub.\n\n"
                        "We typically reply to all project inquiries within 24 business hours.\n\n"
                        "If you would like to book a free 15-minute consultation, reach out to us at contact@genkit.in today!"
                    )
                    inst = format_training_instruction(query, "contact", "contact_info: email", context_text)
                    samples.append({"instruction": inst, "intent": "contact", "output": ans})

    # 4. pricing
    pricing_templates = [
        "website cost", "logo design price", "how much do you charge", "rates for video editing",
        "pricing packages", "starting budget for landing page", "hourly rates"
    ]
    context_text = "Starter landing page package starts at $500, business websites at $1500, e-commerce stores at $3000. Hourly web development rate is $45."
    for temp in pricing_templates:
        for p in PREFIXES:
            for s in SUFFIXES:
                query = f"{p}{temp}{s}".strip()
                ans = (
                    "Reasoning: The user is asking about pricing or service rates. I will present our transparent packages, starting tiers, and customized quotes option.\n\n"
                    "Answer:\nGenkit AI believes in transparent, competitive pricing to deliver outstanding value for startups and businesses.\n\n"
                    "Our starter landing page package starts at $500, business websites at $1500, e-commerce stores at $3000, and custom AI chatbot systems at $4500.\n\n"
                    "For hourly contracts, our rates are $45/hour for web development, $35/hour for UI/UX, and $65/hour for custom AI engineering.\n\n"
                    "Contact us at contact@genkit.in for a custom proposal!"
                )
                inst = format_training_instruction(query, "pricing", "pricing_signal: true", context_text)
                samples.append({"instruction": inst, "intent": "pricing", "output": ans})

    # 5. services
    if "services" in kb:
        for service in kb["services"]:
            name = service.get("service_name", "Service")
            desc = service.get("description", "")
            techs = ", ".join(service.get("technologies", []))
            benefits = ", ".join(service.get("benefits", []))
            turnaround = service.get("turnaround", "2-4 weeks")
            context_text = f"{name} service: {desc} Technologies: {techs}. Benefits: {benefits}. Turnaround: {turnaround}."
            
            service_templates = [
                f"tell me about {name}", f"what is your {name} service", f"do you offer {name}",
                f"benefits of {name}", f"how long does {name} take"
            ]
            for temp in service_templates:
                for p in PREFIXES:
                    for s in SUFFIXES:
                        query = f"{p}{temp}{s}".strip()
                        ans = (
                            f"Reasoning: The user is inquiring about our {name} service. I must explain what we offer, our technologies, and benefits.\n\n"
                            f"Answer:\nGenkit AI offers professional, end-to-end {name} services.\n\n"
                            f"{desc} We leverage standard industry tools including {techs}.\n\n"
                            f"Key features include: {benefits}.\n\n"
                            f"Projects typically have a turnaround timeline of {turnaround}.\n\n"
                            f"Contact us at contact@genkit.in to get started!"
                        )
                        inst = format_training_instruction(query, "services", f"services: {name}", context_text)
                        samples.append({"instruction": inst, "intent": "services", "output": ans})

    unique_samples = {}
    for s in samples:
        inst_clean = s["instruction"].strip().lower()
        if inst_clean not in unique_samples:
            unique_samples[inst_clean] = s
            
    final_samples = list(unique_samples.values())
    random.shuffle(final_samples)
    
    logger.info(f"Generated {len(final_samples)} unique training prompt samples.")
    return final_samples


def main():
    logger.info("Starting Genkit AI Dataset Generator...")
    try:
        kb = load_structured_kb()
        data = generate_samples(kb)
        
        output_file = DATASET_DIR / "dataset.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully wrote {len(data)} training pairs to {output_file}")
    except Exception as e:
        logger.exception("Failed to generate dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
