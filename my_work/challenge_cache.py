"""
Read/query the local challenge cache built by scripts/build_challenge_cache.py.

Cache format (JSONL):
  {"row": int, "image_id": str, "true_label": str, "clean_image_b64": str}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from paths import IMAGENET100_SAMPLES_DIR

DEFAULT_CACHE = IMAGENET100_SAMPLES_DIR / "challenge_cache.jsonl"


def iter_cache(path: Path = DEFAULT_CACHE) -> Iterator[dict]:
    """Yield every record from the cache file, one by one."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_cache(path: Path = DEFAULT_CACHE) -> list[dict]:
    """Load the entire cache into memory as a list of dicts."""
    return list(iter_cache(path))


def cache_stats(path: Path = DEFAULT_CACHE) -> dict:
    """Return quick statistics without loading images into memory."""
    from collections import Counter
    labels: Counter = Counter()
    total = 0
    for record in iter_cache(path):
        labels[record["true_label"]] += 1
        total += 1
    return {
        "total": total,
        "unique_labels": len(labels),
        "top10": labels.most_common(10),
    }


def lookup_by_image_id(image_id: str, path: Path = DEFAULT_CACHE) -> dict | None:
    """Find a single record by image_id (linear scan)."""
    for record in iter_cache(path):
        if record["image_id"] == image_id:
            return record
    return None
