"""
backend/app/rag/loader.py
----------------------------------------------------
Ingests verified Genkit JSON dataset files into structured, metadata-enriched DocumentChunks.
Dedicated domain parsers for services, pricing, company, team, projects, technologies,
faq, policies, portfolio, clients, and contact knowledge bases.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings
from app.core.logger import logger
from app.rag.chunker import DocumentChunk


def extract_keywords(text: str) -> List[str]:
    """Extracts distinctive lowercase alphanumeric keywords."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    stop_words = {
        "and", "the", "for", "with", "that", "this", "from", "are", "our", "you",
        "your", "all", "can", "will", "has", "have", "been", "about", "what", "which",
        "their", "then", "into", "more", "such"
    }
    return list(dict.fromkeys(w for w in words if w not in stop_words))[:25]


def _process_services(data: List[dict], source: str) -> List[DocumentChunk]:
    chunks = []
    service_lines = []

    for idx, item in enumerate(data, 1):
        s_name = item.get("service_name") or item.get("name") or f"Service {idx}"
        desc = item.get("description", "")
        techs = ", ".join(item.get("technologies", [])) if isinstance(item.get("technologies"), list) else str(item.get("technologies", ""))
        benefits = ", ".join(item.get("benefits", [])) if isinstance(item.get("benefits"), list) else str(item.get("benefits", ""))
        turnaround = item.get("turnaround", "")

        service_lines.append(f"- **{s_name}**: {desc} (Tech: {techs} | Turnaround: {turnaround})")

        body_parts = [desc]
        if techs:
            body_parts.append(f"Technologies: {techs}")
        if benefits:
            body_parts.append(f"Key Benefits: {benefits}")
        if turnaround:
            body_parts.append(f"Turnaround: {turnaround}")

        full_text = f"**{s_name}**\n" + "\n".join(body_parts)

        keywords = extract_keywords(f"{s_name} {desc} {techs} {benefits} {turnaround}") + [
            "service", "services", "offer", "offered", "solutions", s_name.lower()
        ]

        chunks.append(
            DocumentChunk(
                id=f"service_{idx}",
                source=source,
                category="Services",
                title=s_name,
                text=full_text,
                keywords=keywords,
                priority=4,
            )
        )

    # Master Overview Chunk
    overview_text = (
        "Genkit offers 6 primary digital and engineering services:\n"
        + "\n".join(service_lines)
    )
    chunks.append(
        DocumentChunk(
            id="services_overview",
            source=source,
            category="Services",
            title="Genkit Services Overview",
            text=overview_text,
            keywords=[
                "services", "service", "offered", "offer", "provide", "solutions",
                "what do you do", "website", "design", "ai", "seo", "video", "branding"
            ],
            priority=5,
        )
    )
    return chunks


def _process_pricing(data: dict, source: str) -> List[DocumentChunk]:
    chunks = []
    currency = data.get("currency", "USD")

    # Packages
    packages = data.get("packages", [])
    package_summaries = []
    for idx, pkg in enumerate(packages, 1):
        p_name = pkg.get("name", "Package")
        price = pkg.get("starting_price", "Custom")
        features = "; ".join(pkg.get("features", [])) if isinstance(pkg.get("features"), list) else str(pkg.get("features", ""))
        timeline = pkg.get("timeline", "")

        pkg_text = f"**{p_name}**: Starting at ${price} {currency} (Timeline: {timeline}). Includes: {features}."
        package_summaries.append(pkg_text)

        chunks.append(
            DocumentChunk(
                id=f"pricing_pkg_{idx}",
                source=source,
                category="Pricing",
                title=p_name,
                text=f"{p_name}: Starts at ${price} {currency}. Timeline: {timeline}. Features: {features}.",
                keywords=extract_keywords(f"{p_name} {price} {features} {timeline}") + ["pricing", "cost", "package", "rate", "website cost", "price"],
                priority=4,
            )
        )

    # Hourly rates
    hourly = data.get("hourly_rates", {})
    if hourly:
        rate_lines = [f"- **{k.replace('_', ' ').title()}**: ${v}/hr" for k, v in hourly.items()]
        hourly_text = "Genkit Hourly Rates:\n" + "\n".join(rate_lines)
        chunks.append(
            DocumentChunk(
                id="pricing_hourly",
                source=source,
                category="Pricing",
                title="Hourly Rates",
                text=hourly_text,
                keywords=["hourly", "rates", "rate", "cost", "per hour", "price", "pricing"],
                priority=3,
            )
        )

    # Master Pricing Overview
    flexible = data.get("flexible_budget", "")
    overview_text = (
        f"Genkit Pricing & Packages ({currency}):\n\n"
        + "\n".join(package_summaries)
        + f"\n\n{flexible}"
    )
    chunks.append(
        DocumentChunk(
            id="pricing_overview",
            source=source,
            category="Pricing",
            title="Genkit Pricing & Packages",
            text=overview_text,
            keywords=["pricing", "price", "cost", "package", "rates", "quote", "how much", "budget", "consultation"],
            priority=5,
        )
    )
    return chunks


def _process_company(data: dict, source: str) -> List[DocumentChunk]:
    chunks = []
    c_name = data.get("company_name", "Genkit")
    founded = data.get("founded", "June 2024")
    founders = ", ".join(data.get("founders", [])) if isinstance(data.get("founders"), list) else str(data.get("founders", "Kishore Kumar, Hari Krishna"))
    team_size = data.get("team_size", "10-15 specialists")
    tagline = data.get("tagline", "From Vision to Digital Reality")
    motto = data.get("motto", "Innovate, Iterate, Impact")
    mission = data.get("mission", "")
    vision = data.get("vision", "")
    model = data.get("operational_model", "Fully remote agency based in India serving global clients.")

    company_overview = (
        f"**{c_name}** was founded in {founded} by {founders}.\n"
        f"- **Tagline**: {tagline}\n"
        f"- **Motto**: {motto}\n"
        f"- **Team Size**: {team_size}\n"
        f"- **Operational Model**: {model}\n"
        f"- **Mission**: {mission}\n"
        f"- **Vision**: {vision}"
    )
    chunks.append(
        DocumentChunk(
            id="company_overview",
            source=source,
            category="Company",
            title="About Genkit",
            text=company_overview,
            keywords=["company", "genkit", "about", "who", "founded", "founders", "kishore", "hari", "team", "mission", "vision", "remote", "what is genkit"],
            priority=5,
        )
    )

    chunks.append(
        DocumentChunk(
            id="company_founders",
            source=source,
            category="Company",
            title="Genkit Founders",
            text=f"Genkit was founded in {founded} by {founders}. The company operates with a remote-first team of {team_size} digital specialists.",
            keywords=["founders", "founder", "kishore", "hari", "who started", "who founded", "created by", "established by"],
            priority=5,
        )
    )

    chunks.append(
        DocumentChunk(
            id="company_mission_vision",
            source=source,
            category="Company",
            title="Mission & Vision",
            text=f"Genkit Mission: {mission}\nVision: {vision}",
            keywords=["mission", "vision", "goal", "purpose", "motto", "values"],
            priority=4,
        )
    )

    return chunks


def _process_technologies(data: dict, source: str) -> List[DocumentChunk]:
    chunks = []
    overview_dict = data.get("tech_stack_overview", {})
    overview_text = overview_dict.get("description", "") if isinstance(overview_dict, dict) else str(overview_dict)

    backend_items = data.get("backend", [])
    frontend_items = data.get("frontend", [])
    database_items = data.get("databases", [])
    creative_items = data.get("creative_tools", [])

    backend_str = ", ".join([x.get("name", "") for x in backend_items]) if isinstance(backend_items, list) else str(backend_items)
    frontend_str = ", ".join([x.get("name", "") for x in frontend_items]) if isinstance(frontend_items, list) else str(frontend_items)
    db_str = ", ".join([x.get("name", "") for x in database_items]) if isinstance(database_items, list) else str(database_items)
    tools_str = ", ".join([x.get("name", "") for x in creative_items]) if isinstance(creative_items, list) else str(creative_items)

    master_tech = (
        f"Genkit Tech Stack Overview: {overview_text}\n\n"
        f"- **Frontend**: {frontend_str}\n"
        f"- **Backend**: {backend_str}\n"
        f"- **Databases**: {db_str}\n"
        f"- **Creative & Design Tools**: {tools_str}"
    )
    chunks.append(
        DocumentChunk(
            id="tech_overview",
            source=source,
            category="Technologies",
            title="Genkit Tech Stack Overview",
            text=master_tech,
            keywords=["tech", "technology", "technologies", "stack", "framework", "languages", "python", "react", "fastapi", "django", "database", "mysql", "databases", "what tech stack"],
            priority=5,
        )
    )

    # Detailed chunks for each tech category
    for cat_name, items in [
        ("Backend Frameworks", backend_items),
        ("Frontend Technologies", frontend_items),
        ("Databases", database_items),
        ("Creative Design Tools", creative_items),
    ]:
        if isinstance(items, list):
            lines = [f"- **{it.get('name', '')}**: {it.get('description', '')}" for it in items if isinstance(it, dict)]
            if lines:
                chunks.append(
                    DocumentChunk(
                        id=f"tech_{cat_name.lower().replace(' ', '_')}",
                        source=source,
                        category="Technologies",
                        title=cat_name,
                        text=f"Genkit {cat_name}:\n" + "\n".join(lines),
                        keywords=extract_keywords(" ".join(lines)) + [cat_name.lower(), "tech", "stack"],
                        priority=4,
                    )
                )

    return chunks


def _process_contact(data: dict, source: str) -> List[DocumentChunk]:
    chunks = []
    email = data.get("email", "genkit.tech@gmail.com")
    website = data.get("website", "https://www.genkit.in")
    form = data.get("contact_form", "https://www.genkit.in/contact")
    channels_list = data.get("channels", [])
    resp_time = data.get("response_time", "Within 24 business hours")
    consultations = data.get("consultations", "Free 15-minute scoping calls available at genkit.in.")

    channel_lines = []
    if isinstance(channels_list, list):
        for ch in channels_list:
            if isinstance(ch, dict):
                channel_lines.append(f"- **{ch.get('type', '')}**: {ch.get('address') or ch.get('handle', '')} ({ch.get('best_for', '')})")

    contact_text = (
        f"You can contact Genkit AI easily:\n"
        f"- **Email**: {email}\n"
        f"- **Website**: [{website}]({website})\n"
        f"- **Contact Form**: {form}\n"
        f"- **Response Time**: {resp_time}\n"
        f"- **Consultation**: {consultations}\n\n"
        f"**Active Channels**:\n"
        + "\n".join(channel_lines)
    )
    chunks.append(
        DocumentChunk(
            id="contact_info",
            source=source,
            category="Contact",
            title="Contact Genkit",
            text=contact_text,
            keywords=["contact", "email", "website", "reach", "hire", "phone", "support", "quote", "form", "genkit.tech@gmail.com", "instagram", "github", "how to contact"],
            priority=5,
        )
    )
    return chunks


def _process_projects(data: List[dict], source: str) -> List[DocumentChunk]:
    chunks = []
    for idx, item in enumerate(data, 1):
        p_name = item.get("project_name") or f"Project {idx}"
        svc = item.get("service", "")
        desc = item.get("description", "")
        tech = item.get("technology", [])
        tech_str = ", ".join(tech) if isinstance(tech, list) else str(tech)
        impact = item.get("impact", "")

        full_text = (
            f"**Project: {p_name}**\n"
            f"- **Service Area**: {svc}\n"
            f"- **Description**: {desc}\n"
            f"- **Technologies**: {tech_str}\n"
            f"- **Business Impact**: {impact}"
        )
        chunks.append(
            DocumentChunk(
                id=f"project_{idx}",
                source=source,
                category="Portfolio",
                title=p_name,
                text=full_text,
                keywords=extract_keywords(f"{p_name} {svc} {desc} {tech_str} {impact}") + ["project", "case study", "portfolio", "delivered"],
                priority=4,
            )
        )
    return chunks


def _process_team(data: List[dict], source: str) -> List[DocumentChunk]:
    chunks = []
    member_summaries = []
    for idx, item in enumerate(data, 1):
        name = item.get("name", "")
        role = item.get("role", "")
        spec = item.get("specialty", "")
        bg = item.get("background", "")

        member_summaries.append(f"- **{name}** ({role}): {spec}")

        chunks.append(
            DocumentChunk(
                id=f"team_{idx}",
                source=source,
                category="Team",
                title=name,
                text=f"**{name}** — {role}\nSpecialty: {spec}\nBackground: {bg}",
                keywords=extract_keywords(f"{name} {role} {spec} {bg}") + ["team", "engineer", "designer", "developer", name.lower()],
                priority=3,
            )
        )

    # Master Team Overview Chunk
    chunks.append(
        DocumentChunk(
            id="team_overview",
            source=source,
            category="Team",
            title="Genkit Team Overview",
            text="Genkit core team members and leadership:\n" + "\n".join(member_summaries),
            keywords=["team", "members", "staff", "engineers", "who works", "leadership", "developers", "designers"],
            priority=5,
        )
    )
    return chunks


def _process_faq(data: List[dict], source: str) -> List[DocumentChunk]:
    chunks = []
    for idx, item in enumerate(data, 1):
        q = item.get("question", "")
        a = item.get("answer", "")
        if q and a:
            chunks.append(
                DocumentChunk(
                    id=f"faq_{idx}",
                    source=source,
                    category="FAQ",
                    title=q,
                    text=a.strip(),
                    keywords=extract_keywords(f"{q} {a}") + ["faq", "question", "how", "why", "do you"],
                    priority=4,
                )
            )
    return chunks


def _process_policies(data: dict, source: str) -> List[DocumentChunk]:
    chunks = []
    for pol_key, pol_val in data.items():
        title = pol_key.replace("_", " ").title()
        if isinstance(pol_val, dict):
            body = " | ".join([f"{k.title()}: {v}" for k, v in pol_val.items()])
        else:
            body = str(pol_val)

        chunks.append(
            DocumentChunk(
                id=f"policy_{pol_key}",
                source=source,
                category="Policies",
                title=title,
                text=f"**{title}**\n{body}",
                keywords=extract_keywords(f"{title} {body}") + ["policy", "terms", "support", "refund", "nda", "revision", "privacy"],
                priority=4,
            )
        )
    return chunks


def _process_portfolio(data: dict, source: str) -> List[DocumentChunk]:
    chunks = []
    philosophy = data.get("design_philosophy", "")
    if philosophy:
        chunks.append(
            DocumentChunk(
                id="portfolio_philosophy",
                source=source,
                category="Portfolio",
                title="Design Philosophy",
                text=f"**Genkit Design Philosophy**: {philosophy}",
                keywords=["design", "philosophy", "ui", "ux", "creative", "approach"],
                priority=3,
            )
        )

    case_study = data.get("featured_case_study", {})
    if isinstance(case_study, dict):
        cs_title = case_study.get("title", "Featured Case Study")
        cs_challenge = case_study.get("challenge", "")
        cs_solution = case_study.get("solution", "")
        cs_result = case_study.get("result", "")
        cs_text = (
            f"**Case Study: {cs_title}**\n"
            f"- **Challenge**: {cs_challenge}\n"
            f"- **Solution**: {cs_solution}\n"
            f"- **Result**: {cs_result}"
        )
        chunks.append(
            DocumentChunk(
                id="portfolio_case_study",
                source=source,
                category="Portfolio",
                title=cs_title,
                text=cs_text,
                keywords=extract_keywords(f"{cs_title} {cs_challenge} {cs_solution} {cs_result}") + ["case study", "portfolio", "work", "example"],
                priority=4,
            )
        )

    return chunks


def _process_clients(data: dict, source: str) -> List[DocumentChunk]:
    chunks = []
    industries = data.get("industries_served", [])
    if isinstance(industries, list):
        ind_str = ", ".join(industries)
        chunks.append(
            DocumentChunk(
                id="clients_industries",
                source=source,
                category="Clients",
                title="Industries Served",
                text=f"Genkit provides digital services to diverse industries: {ind_str}.",
                keywords=extract_keywords(ind_str) + ["industries", "clients", "sectors", "who do you work with"],
                priority=3,
            )
        )

    testimonials = data.get("testimonials", [])
    if isinstance(testimonials, list):
        for idx, t in enumerate(testimonials, 1):
            if isinstance(t, dict):
                client = t.get("client", "")
                comp = t.get("company", "")
                proj = t.get("project", "")
                fb = t.get("feedback", "")
                chunks.append(
                    DocumentChunk(
                        id=f"client_testimonial_{idx}",
                        source=source,
                        category="Clients",
                        title=f"Testimonial from {client} ({comp})",
                        text=f"**{client}** ({comp}) on *{proj}*:\n\"{fb}\"",
                        keywords=extract_keywords(f"{client} {comp} {proj} {fb}") + ["testimonial", "review", "feedback", "rating", "client"],
                        priority=3,
                    )
                )

    return chunks


def load_domain_chunks() -> List[DocumentChunk]:
    """Loads curated knowledge files from backend/datasets/ and converts them to rich DocumentChunks."""
    dataset_dir = settings.DATASET_DIR
    chunks: List[DocumentChunk] = []

    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return chunks

    all_files = sorted(dataset_dir.glob("*.json"))
    domain_files = [f for f in all_files if f.name.lower() not in ("dataset.json", "dataset_raw.json")]
    target_files = domain_files if domain_files else all_files

    for json_file in target_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            source = json_file.stem.lower()

            if source == "services" and isinstance(data, list):
                file_chunks = _process_services(data, source)
            elif source == "pricing" and isinstance(data, dict):
                file_chunks = _process_pricing(data, source)
            elif source == "company" and isinstance(data, dict):
                file_chunks = _process_company(data, source)
            elif source == "technologies" and isinstance(data, dict):
                file_chunks = _process_technologies(data, source)
            elif source == "contact" and isinstance(data, dict):
                file_chunks = _process_contact(data, source)
            elif source == "projects" and isinstance(data, list):
                file_chunks = _process_projects(data, source)
            elif source == "team" and isinstance(data, list):
                file_chunks = _process_team(data, source)
            elif source == "faq" and isinstance(data, list):
                file_chunks = _process_faq(data, source)
            elif source == "policies" and isinstance(data, dict):
                file_chunks = _process_policies(data, source)
            elif source == "portfolio" and isinstance(data, dict):
                file_chunks = _process_portfolio(data, source)
            elif source == "clients" and isinstance(data, dict):
                file_chunks = _process_clients(data, source)
            else:
                file_chunks = []

            chunks.extend(file_chunks)
            logger.info(f"Loaded {json_file.name} — Added {len(file_chunks)} chunks (Total: {len(chunks)})")
        except Exception as e:
            logger.error(f"Error reading {json_file.name}: {e}")

    logger.info(f"Total knowledge base documents indexed: {len(chunks):,}")
    return chunks
