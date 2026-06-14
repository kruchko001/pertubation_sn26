"""Shared paths for my_work scripts and tools."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PERTURB = ROOT.parent / "Perturb"
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"

DOG_IMAGES_DIR = DATA / "dog_images"
PROMPT_SAMPLES_DIR = DATA / "prompt_samples"
PEXELS_COUNTS_JSON = OUTPUTS / "pexels_photo_counts.json"
IMAGES_PER_PROMPT_CSV = OUTPUTS / "images_per_prompt.csv"
