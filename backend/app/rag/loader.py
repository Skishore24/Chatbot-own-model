"""
backend/app/rag/loader.py
----------------------------------------------------
Ingests verified Genkit JSON dataset files into structured, metadata-enriched DocumentChunks.
Handles lists, dictionaries, nested arrays, project schemas, and QA pairs.
"""

import json
import re
from pathlib import Path
from typing import Any, List

from app.core.config import settings
from app.core.logger import logger
from app.rag.chunker import DocumentChunk


def extract_keywords(text: str) -> List[str]:
    """Extracts distinctive lowercase alphanumeric keywords."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    stop_words = {
        "and", "the", "for", "with", "that", "this", "from", "are", "our", "you",
        "your", "all", "can", "will", "has", "have", "been", "about", "what", "which"
    }
    return list(dict.fromkeys(w for w in words if w not in stop_words))[:15]


def _process_item(item: Any, source: str, prefix: str = "", idx: int = 0) -> List[DocumentChunk]:
    """Recursively processes JSON items into DocumentChunks."""
    chunks: List[DocumentChunk] = []

    if isinstance(item, dict):
        # Case 1: Named entity, project, or QA pair
        name = (
            item.get("name")
            or item.get("project_name")
            or item.get("title")
            or item.get("question")
            or item.get("instruction")
            or item.get("service")
        )

        desc_parts = []
        for field in ["description", "content", "answer", "output", "details", "impact", "summary", "role", "specialty", "background", "bio"]:
            if field in item and item[field]:
                desc_parts.append(str(item[field]).strip())

        if "technology" in item:
            tech = item["technology"]
            tech_str = ", ".join(tech) if isinstance(tech, list) else str(tech)
            desc_parts.append(f"Technologies: {tech_str}")

        if name and desc_parts:
            title = str(name).strip()
            body = " | ".join(desc_parts).strip()
            chunk_id = f"{source}_{prefix}_{idx}" if prefix else f"{source}_{idx}"
            category = item.get("service") or item.get("category") or prefix or source
            keywords = item.get("keywords") or extract_keywords(f"{title} {body}")
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    source=source,
                    category=category,
                    title=title,
                    text=f"{title}: {body}",
                    keywords=keywords,
                    priority=item.get("priority", 1),
                )
            )
            return chunks

        # Case 2: Dict containing nested lists or sub-dicts (e.g. technologies.json)
        has_nested = False
        for k, v in item.items():
            if isinstance(v, list):
                has_nested = True
                for sub_idx, sub_item in enumerate(v):
                    chunks.extend(_process_item(sub_item, source, prefix=k, idx=sub_idx + 1))
            elif isinstance(v, dict):
                has_nested = True
                chunks.extend(_process_item(v, source, prefix=k, idx=1))

        if not has_nested:
            # Flat key-value dict (e.g. company.json)
            for k, v in item.items():
                title = k.replace("_", " ").title()
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    body = ", ".join(v).strip()
                    if body:
                        chunks.append(
                            DocumentChunk(
                                id=f"{source}_{k}",
                                source=source,
                                category=source,
                                title=title,
                                text=f"{title}: {body}",
                                keywords=extract_keywords(f"{title} {body}") + [k.lower(), title.lower()],
                                priority=2 if k.lower() in ("founders", "mission", "vision", "services", "technologies") else 1,
                            )
                        )
                elif isinstance(v, str) and len(v.strip()) > 3:
                    body = v.strip()
                    chunks.append(
                        DocumentChunk(
                            id=f"{source}_{k}",
                            source=source,
                            category=source,
                            title=title,
                            text=f"{title}: {body}",
                            keywords=extract_keywords(f"{title} {body}") + [k.lower(), title.lower()],
                            priority=2 if k.lower() in ("mission", "vision", "operational_model", "tagline") else 1,
                        )
                    )


    elif isinstance(item, list):
        for sub_idx, sub_item in enumerate(item):
            chunks.extend(_process_item(sub_item, source, prefix=prefix, idx=sub_idx + 1))

    elif isinstance(item, str) and len(item.strip()) > 10:
        chunks.append(
            DocumentChunk(
                id=f"{source}_{prefix}_{idx}",
                source=source,
                category=prefix or source,
                title=f"{source.title()} Knowledge",
                text=item.strip(),
                keywords=extract_keywords(item),
                priority=1,
            )
        )

    return chunks


def load_domain_chunks() -> List[DocumentChunk]:
    """Loads curated knowledge files from backend/datasets/ and converts them to DocumentChunks."""
    dataset_dir = settings.DATASET_DIR
    chunks: List[DocumentChunk] = []

    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return chunks

    # Prioritize domain knowledge files, skipping raw instruction training corpus (dataset.json)
    all_files = sorted(dataset_dir.glob("*.json"))
    domain_files = [f for f in all_files if f.name.lower() not in ("dataset.json", "dataset_raw.json")]
    target_files = domain_files if domain_files else all_files

    for json_file in target_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            source = json_file.stem.lower()
            file_chunks = _process_item(data, source)
            chunks.extend(file_chunks)
            logger.info(f"Loaded {json_file.name} — Added {len(file_chunks)} chunks (Total: {len(chunks)})")
        except Exception as e:
            logger.error(f"Error reading {json_file.name}: {e}")

    logger.info(f"Total knowledge base documents indexed: {len(chunks):,}")
    return chunks

