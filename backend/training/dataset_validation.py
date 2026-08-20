"""
backend/training/dataset_validation.py
----------------------------------------------------
Dataset audit and schema validator for Genkit AI V6.
Scans all JSON files in backend/datasets/, validates format, checks duplicates,
and computes vocabulary & length statistics.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Set

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger


def validate_datasets() -> dict:
    """Validates all JSON files in datasets directory and returns report statistics."""
    dataset_dir = settings.DATASET_DIR
    files_checked = []
    total_samples = 0
    duplicate_count = 0
    seen_texts: Set[str] = set()
    lengths: List[int] = []
    category_counts: Dict[str, int] = {}
    errors: List[str] = []

    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return {"error": "Dataset dir missing"}

    for json_file in sorted(dataset_dir.glob("*.json")):
        files_checked.append(json_file.name)
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_samples = 0
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    sample_text = ""
                    if isinstance(item, dict):
                        for k in ["instruction", "question", "text", "description", "content", "answer", "output", "role", "specialty", "background", "name"]:
                            val = item.get(k)
                            if val and isinstance(val, str):
                                sample_text += " " + val.strip()
                    elif isinstance(item, str):
                        sample_text = item.strip()

                    sample_text = sample_text.strip()
                    if sample_text:
                        file_samples += 1
                        total_samples += 1
                        lengths.append(len(sample_text.split()))

                        if sample_text.lower() in seen_texts:
                            duplicate_count += 1
                        else:
                            seen_texts.add(sample_text.lower())

            elif isinstance(data, dict):
                for k, v in data.items():
                    text = f"{k}: {v}" if isinstance(v, str) else json.dumps(v)
                    file_samples += 1
                    total_samples += 1
                    lengths.append(len(text.split()))
                    seen_texts.add(text.lower())

            category_counts[json_file.stem] = file_samples
        except Exception as e:
            errors.append(f"{json_file.name}: {str(e)}")

    avg_len = sum(lengths) / max(len(lengths), 1)
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0

    report = {
        "status": "valid" if not errors else "errors_found",
        "files_checked": files_checked,
        "total_samples": total_samples,
        "unique_samples": len(seen_texts),
        "duplicates": duplicate_count,
        "avg_word_length": round(avg_len, 2),
        "max_word_length": max_len,
        "min_word_length": min_len,
        "categories": category_counts,
        "errors": errors,
    }

    print("\n" + "=" * 60)
    print("  GENKIT AI DATASET AUDIT REPORT")
    print("=" * 60)
    print(f"  Files Checked     : {len(files_checked)}")
    print(f"  Total Samples     : {total_samples:,}")
    print(f"  Unique Samples    : {len(seen_texts):,}")
    print(f"  Duplicates        : {duplicate_count:,}")
    print(f"  Avg Word Length   : {avg_len:.1f} words")
    print(f"  Max Word Length   : {max_len} words")
    print("-" * 60)
    for cat, count in category_counts.items():
        print(f"  - {cat:<18}: {count:,} samples")
    print("=" * 60 + "\n")

    return report


if __name__ == "__main__":
    validate_datasets()
