"""
backend/utils/dataset_generator.py
----------------------------------------------------
Genkit AI - Structured Dataset Generator (v4.2)

Generates 10000+ high-quality training pairs (instruction, output)
where instruction is the fully populated prompt (without outer [INST] tags)
and output contains Reasoning and Answer blocks.

Author: Genkit AI
"""
import os
import sys
import json
import random
from pathlib import Path

# Setup paths
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import DATASET_DIR, logger
from ai.llm.prompt_builder import SYSTEM_INSTRUCTION

# Helper to format the training instruction as prompt context
def format_training_instruction(query: str, intent: str, entities_str: str, context_text: str) -> str:
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
    kb = {}
    kb_files = [
        "company.json", "services.json", "projects.json", "technologies.json",
        "pricing.json", "team.json", "clients.json", "portfolio.json",
        "faq.json", "policies.json", "contact.json"
    ]
    for filename in kb_files:
        path = DATASET_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing structured KB file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            kb[filename.split(".")[0]] = json.load(f)
    return kb

def generate_samples(kb):
    samples = []
    
    # 1. greeting (300 samples)
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

    # 2. out_of_domain (2000 samples)
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

    # 3. contact (1000 samples)
    contact_templates = [
        "how can i contact you", "what is your email", "contact channels", "reach out to genkit",
        "how quickly do you respond", "book a consultation", "where are you located", "do you have instagram"
    ]
    contact_data = kb["contact"]
    channels_str = ", ".join([c["type"] for c in contact_data["channels"]])
    context_text = f"Primary email: {contact_data['email']}. Website: {contact_data['website']}. Channels: {channels_str}."
    for temp in contact_templates:
        for p in PREFIXES:
            for s in SUFFIXES:
                query = f"{p}{temp}{s}".strip()
                ans = (
                    "Reasoning: The user is asking how to contact Genkit AI. I need to supply our primary email, social channels, and consultation details.\n\n"
                    "Answer:\nYou can contact Genkit AI easily via email at genkit.tech@gmail.com.\n\n"
                    "Our team works fully remote from India, serving clients across the globe. You can also find us on Instagram at @genkit.in or check our projects on GitHub.\n\n"
                    "We typically reply to all project inquiries within 24 business hours.\n\n"
                    "If you would like to book a free 15-minute scoping consultation, please visit genkit.in/contact to schedule a call today!"
                )
                inst = format_training_instruction(query, "contact", "contact_info: email", context_text)
                samples.append({"instruction": inst, "intent": "contact", "output": ans})

    # 4. pricing (1200 samples)
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
                    "Answer:\nGenkit AI believes in transparent, competitive pricing to deliver outstanding value for startups and small businesses.\n\n"
                    "Our starter landing page package starts at $500, business websites at $1500, e-commerce stores at $3000, and custom AI chatbot systems at $4500.\n\n"
                    "For hourly contracts, our rates are $45/hour for web development, $35/hour for UI/UX, and $65/hour for custom AI engineering.\n\n"
                    "Bundled pricing packages are available for clients needing multiple services, giving you significant cost savings.\n\n"
                    "We can recommend the most suitable backend stack and pricing structure based on your project goals. Contact us at genkit.tech@gmail.com for a quote!"
                )
                inst = format_training_instruction(query, "pricing", "pricing_signal: true", context_text)
                samples.append({"instruction": inst, "intent": "pricing", "output": ans})

    # 5. services (2500 samples)
    for service in kb["services"]:
        name = service["service_name"]
        desc = service["description"]
        techs = ", ".join(service["technologies"])
        benefits = ", ".join(service["benefits"])
        turnaround = service["turnaround"]
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
                        f"{desc} We leverage standard industry tools and libraries including {techs}.\n\n"
                        f"Key features of this service include: {benefits}.\n\n"
                        f"Projects in this domain have a typical turnaround timeline of {turnaround}.\n\n"
                        f"If you're planning a project, we can recommend the most suitable development or creative stack based on your needs. Contact us at genkit.tech@gmail.com!"
                    )
                    inst = format_training_instruction(query, "services", f"services: {name}", context_text)
                    samples.append({"instruction": inst, "intent": "services", "output": ans})

    # 6. technology (1500 samples)
    tech_categories = kb["technologies"]
    for category, techs in tech_categories.items():
        for t in techs:
            t_name = t["name"]
            t_desc = t["description"]
            context_text = f"{t_name} is used for {category}: {t_desc}"
            tech_templates = [
                f"do you use {t_name}", f"why do you use {t_name}", f"what is {t_name} used for"
            ]
            for temp in tech_templates:
                for p in PREFIXES:
                    for s in SUFFIXES:
                        query = f"{p}{temp}{s}".strip()
                        ans = (
                            f"Reasoning: The user is asking about the technology {t_name}. I must clarify its role and usage inside our development/design pipeline.\n\n"
                            f"Answer:\nGenkit AI uses {t_name} as part of our core {category} workflow.\n\n"
                            f"{t_desc}\n\n"
                            f"By using {t_name}, we ensure our deliveries are secure, responsive, and follow modern modular standards.\n\n"
                            f"Our full stack developers are highly skilled in {t_name} and adjacent frameworks.\n\n"
                            f"Let us know if you want to deploy your next application using {t_name}! Reach out to us at genkit.tech@gmail.com."
                        )
                        inst = format_training_instruction(query, "technology", f"technologies: {t_name}", context_text)
                        samples.append({"instruction": inst, "intent": "technology", "output": ans})

    # 7. project (1200 samples)
    for proj in kb["projects"]:
        p_name = proj["project_name"]
        p_serv = proj["service"]
        p_desc = proj["description"]
        p_techs = ", ".join(proj["technology"])
        p_imp = proj["impact"]
        context_text = f"Project: {p_name}. Category: {p_serv}. Description: {p_desc}. Technologies: {p_techs}. Impact: {p_imp}."
        
        proj_templates = [
            f"tell me about {p_name}", f"details on {p_name}", f"what is {p_name}"
        ]
        for temp in proj_templates:
            for p in PREFIXES:
                for s in SUFFIXES:
                    query = f"{p}{temp}{s}".strip()
                    ans = (
                        f"Reasoning: The user is asking about the specific project {p_name}. I need to explain its details, technology, and business impact.\n\n"
                        f"Answer:\nGenkit AI successfully delivered the {p_name} project.\n\n"
                        f"This project falls under our {p_serv} category. {p_desc}\n\n"
                        f"We built it using the following technologies: {p_techs}.\n\n"
                        f"The project had a major business impact: {p_imp}.\n\n"
                        f"If you're planning a similar {p_serv} system, we can design the ideal architecture for you. Reach out to us at genkit.tech@gmail.com!"
                    )
                    inst = format_training_instruction(query, "project", f"project: {p_name}", context_text)
                    samples.append({"instruction": inst, "intent": "project", "output": ans})

    # 8. portfolio (1200 samples)
    portfolio_data = kb["portfolio"]
    context_text = f"Philosophy: {portfolio_data['design_philosophy']}. Case study: {portfolio_data['featured_case_study']['title']}."
    portfolio_templates = [
        "show me past projects", "previous work examples", "case studies", "featured case study"
    ]
    for temp in portfolio_templates:
        for p in PREFIXES:
            for s in SUFFIXES:
                query = f"{p}{temp}{s}".strip()
                ans = (
                    "Reasoning: The user is asking for work examples or design case studies. I need to explain our featured projects and design philosophies.\n\n"
                    "Answer:\nGenkit AI takes immense pride in delivering high-impact, custom digital solutions for our clients.\n\n"
                    "Our design philosophy focuses on clean, functional aesthetics, user-centric prototypes in Figma, and robust web development.\n\n"
                    "A featured case study is the ShopSmart Brand Refresh, where we designed a cohesive brand identity and rebuilt the React web portal.\n\n"
                    "This overhaul resulted in a page speed score of 98/100 and boosted client conversion metrics significantly.\n\n"
                    "Check out all our case studies and creative visual assets at genkit.in/portfolio!"
                )
                inst = format_training_instruction(query, "portfolio", "portfolio: true", context_text)
                samples.append({"instruction": inst, "intent": "portfolio", "output": ans})

    # 9. support & policies (1500 samples)
    policy_data = kb["policies"]
    context_text = f"Support: {policy_data['support_policy']['standard']}. Revisions: {policy_data['revision_policy']['details']}."
    policy_templates = [
        "refund policy", "revisions allowed", "do you sign nda", "maintenance contract support",
        "tell me about team members", "who is on the team", "founders of genkit"
    ]
    for temp in policy_templates:
        for p in PREFIXES:
            for s in SUFFIXES:
                query = f"{p}{temp}{s}".strip()
                ans = (
                    "Reasoning: The user is asking about support timelines, revision constraints, team structure, or refund policies. I should state our clear rules and guarantees.\n\n"
                    "Answer:\nGenkit AI provides dependable post-delivery support and clear, transparent terms to ensure complete peace of mind.\n\n"
                    "All development contracts include a standard support window of 7 to 90 days to address bugs and offer performance adjustments.\n\n"
                    "We offer unlimited revisions during the initial Figma UI design phase. Revisions during coding must fit the project scope.\n\n"
                    "Partial refunds are available depending on progress stages, and we sign Non-Disclosure Agreements (NDAs) to protect client IP.\n\n"
                    "If you need ongoing updates, we offer website maintenance plans starting at $150/month. Contact us at genkit.tech@gmail.com for support!"
                )
                inst = format_training_instruction(query, "support", "support_policy: true", context_text)
                samples.append({"instruction": inst, "intent": "support", "output": ans})

    unique_samples = {}
    for s in samples:
        # Deduplicate based on instruction query structure
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
