"""Shared paths for my_work scripts and tools."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"

IMAGENET100_SAMPLES_DIR = DATA / "imagenet100_samples"
