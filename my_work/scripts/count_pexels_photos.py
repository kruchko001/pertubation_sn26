#!/usr/bin/env python3
"""Count Pexels total_results per Perturb prompt and sum by category."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from env import load_env, pexels_api_key
from paths import IMAGES_PER_PROMPT_CSV, PERTURB, PEXELS_COUNTS_JSON

if str(PERTURB) not in sys.path:
    sys.path.insert(0, str(PERTURB))

from perturbnet.constants import IMAGE_ENDPOINT, PROMPTS

CATEGORIES: dict[str, tuple[str, ...]] = {
    "Animals": (
        "dog", "cat", "bird", "fish", "snake", "frog", "butterfly",
        "spider", "crab", "jellyfish", "monkey", "hamster", "rabbit",
        "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear",
    ),
    "Vehicles": (
        "sports car", "truck", "bus", "motorcycle", "bicycle", "airplane",
        "helicopter", "sailboat", "canoe", "train",
    ),
    "Food": (
        "banana", "strawberry", "orange", "broccoli", "mushroom", "pizza",
        "cheeseburger", "ice cream", "coffee mug", "wine bottle",
    ),
    "Everyday Objects": (
        "chair", "lamp", "clock", "backpack", "umbrella", "sunglasses",
        "shoe", "hat", "vase", "television",
    ),
    "Electronics & Instruments": (
        "keyboard", "mouse", "camera", "guitar", "drum", "violin",
        "telescope", "microscope",
    ),
    "Sports & Recreation": (
        "soccer ball", "basketball", "tennis ball", "baseball bat",
        "skateboard", "surfboard", "parachute",
    ),
}

PROMPT_TO_CATEGORY = {
    prompt: category
    for category, prompts in CATEGORIES.items()
    for prompt in prompts
}


def fetch_total(session: requests.Session, endpoint: str, api_key: str, prompt: str) -> int:
    response = session.get(
        endpoint,
        params={"query": prompt, "page": 1, "per_page": 1},
        headers={"Authorization": api_key},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    total = data.get("total_results", 0) if isinstance(data, dict) else 0
    return int(total) if isinstance(total, (int, float)) else 0


def main() -> int:
    load_env()
    api_key = pexels_api_key()
    if not api_key:
        print("Missing PERTURB_PEXELS_API_KEY in my_work/.env", file=sys.stderr)
        return 1

    session = requests.Session()
    by_prompt: dict[str, dict] = {}

    print(f"Querying Pexels for {len(PROMPTS)} prompts...")
    for index, prompt in enumerate(PROMPTS, start=1):
        print(f"  [{index}/{len(PROMPTS)}] {prompt!r}", flush=True)
        try:
            total = fetch_total(session, IMAGE_ENDPOINT, api_key, prompt)
            by_prompt[prompt] = {
                "category": PROMPT_TO_CATEGORY.get(prompt, "Unknown"),
                "pexels_total": total,
            }
        except Exception as exc:
            by_prompt[prompt] = {
                "category": PROMPT_TO_CATEGORY.get(prompt, "Unknown"),
                "pexels_total": None,
                "error": str(exc),
            }
        time.sleep(0.2)

    by_category: dict[str, dict] = {}
    for category in CATEGORIES:
        prompts = CATEGORIES[category]
        totals = [by_prompt[p]["pexels_total"] for p in prompts if by_prompt[p].get("pexels_total") is not None]
        by_category[category] = {
            "prompt_count": len(prompts),
            "pexels_total_sum": sum(totals),
            "pexels_total_avg": round(sum(totals) / len(totals), 1) if totals else 0,
            "pexels_total_min": min(totals) if totals else 0,
            "pexels_total_max": max(totals) if totals else 0,
            "prompts": {p: by_prompt[p] for p in prompts},
        }

    grand_pexels = sum(c["pexels_total_sum"] for c in by_category.values())

    report = {
        "note": (
            "pexels_total = Pexels search total_results for each prompt. "
            "The Perturb validator picks a random page and random photo from this source on each fetch."
        ),
        "grand_totals": {
            "prompt_count": len(PROMPTS),
            "pexels_total_sum": grand_pexels,
        },
        "by_category": by_category,
        "by_prompt": by_prompt,
    }

    PEXELS_COUNTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PEXELS_COUNTS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with IMAGES_PER_PROMPT_CSV.open("w", encoding="utf-8") as handle:
        handle.write("prompt,category,pexels_total\n")
        for prompt in PROMPTS:
            row = by_prompt[prompt]
            total = row.get("pexels_total")
            if total is not None:
                handle.write(f"{prompt},{row['category']},{total}\n")

    print("\n=== Images per category (Pexels total_results) ===\n")
    header = f"{'Category':<28} {'Prompts':>7} {'Sum':>12} {'Avg/prompt':>12} {'Min':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))
    for category in CATEGORIES:
        c = by_category[category]
        print(
            f"{category:<28} {c['prompt_count']:>7} {c['pexels_total_sum']:>12,} "
            f"{c['pexels_total_avg']:>12,.1f} {c['pexels_total_min']:>8,} {c['pexels_total_max']:>8,}"
        )
    print("-" * len(header))
    print(f"{'ALL':<28} {len(PROMPTS):>7} {grand_pexels:>12,} {grand_pexels / len(PROMPTS):>12,.1f}")
    print(f"\nJSON: {PEXELS_COUNTS_JSON}")
    print(f"CSV:  {IMAGES_PER_PROMPT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
