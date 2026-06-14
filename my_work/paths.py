"""Shared paths for my_work scripts and tools."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PERTURB = ROOT.parent / "Perturb"
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"

PROMPT_IMAGES_DIR = DATA / "prompt_images"
# Legacy path kept for older downloads
DOG_IMAGES_DIR = DATA / "dog_images"
PROMPT_SAMPLES_DIR = DATA / "prompt_samples"
PEXELS_COUNTS_JSON = OUTPUTS / "pexels_photo_counts.json"
IMAGES_PER_PROMPT_CSV = OUTPUTS / "images_per_prompt.csv"


def slugify_prompt(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.strip().lower()).strip("_")


def prompt_images_dir(prompt: str) -> Path:
    return PROMPT_IMAGES_DIR / slugify_prompt(prompt)
