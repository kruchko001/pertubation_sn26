#!/usr/bin/env python3
"""
Prepare ImageNet-100 train or validation split for generator training.

Mirrors the validator's clean-image path (JPEG q=95) and EfficientNetV2-L
inference, then materialises local .npy tensors for fast training.

Pipeline (4 steps):
  1. save_images.py      HF image -> JPEG q=95 -> {row:07d}.b64
  2. run_inference.py    .b64 -> EfficientNet logits -> {row:07d}.json
  3. save_decoded_npy.py .b64 -> float32 CHW [0,1] -> {row:07d}.npy
  4. build_indexes.py    .npy/.json -> consolidated shape + label caches

Output layout (train split shown; validation uses imagenet100_val_*):
  data/imagenet100_samples/{row:07d}.b64   validator-exact clean (base64 text)
  data/imagenet100_samples/{row:07d}.json  EfficientNet logits (true label = argmax)
  data/imagenet100_samples/{row:07d}.npy   float32 CHW [0,1] at native resolution
  data/imagenet100_shapes.json              {version, shapes: {row: [H,W]}}
  data/imagenet100_true_labels.json         {version, labels: {row: idx}}

Usage (from my_work/):
  python scripts/prepare_dataset.py --status
  python scripts/prepare_dataset.py --split validation --workers 8
  python scripts/prepare_dataset.py --split train --resume
  python scripts/prepare_dataset.py --split validation --indexes-only
  python scripts/prepare_dataset.py --split train --limit 1000   # smoke test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MY_WORK = Path(__file__).resolve().parent.parent
SCRIPTS = MY_WORK / "scripts"

if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from paths import (  # noqa: E402
    IMAGENET100_LABELS_CACHE,
    IMAGENET100_SAMPLES_DIR,
    IMAGENET100_SHAPES_CACHE,
    IMAGENET100_VAL_LABELS_CACHE,
    IMAGENET100_VAL_SAMPLES_DIR,
    IMAGENET100_VAL_SHAPES_CACHE,
)
from perturb_mirror.constants import IMAGENET100_REPO_ID  # noqa: E402

# Expected row counts for the HF ImageNet-100 splits (used by --status only).
_EXPECTED_ROWS = {"train": 126_689, "validation": 5_000}

SPLIT_CONFIG = {
    "train": {
        "hf_split": "train",
        "samples_dir": IMAGENET100_SAMPLES_DIR,
        "shape_cache": IMAGENET100_SHAPES_CACHE,
        "label_cache": IMAGENET100_LABELS_CACHE,
    },
    "validation": {
        "hf_split": "validation",
        "samples_dir": IMAGENET100_VAL_SAMPLES_DIR,
        "shape_cache": IMAGENET100_VAL_SHAPES_CACHE,
        "label_cache": IMAGENET100_VAL_LABELS_CACHE,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full data prep for train or validation split")
    p.add_argument(
        "--split",
        choices=["train", "validation"],
        default=None,
        help="HF split to prepare (required unless --status)",
    )
    p.add_argument("--workers", type=int, default=8, help="Threads for b64/npy encode")
    p.add_argument("--batch-size", type=int, default=32, help="EfficientNet inference batch size")
    p.add_argument("--inference-workers", type=int, default=4, help="DataLoader workers for inference")
    p.add_argument("--limit", type=int, default=0, help="Max rows (0 = full split)")
    p.add_argument("--resume", action="store_true", help="Skip files that already exist")
    p.add_argument("--indexes-only", action="store_true", help="Only rebuild shape/label caches")
    p.add_argument("--skip-indexes", action="store_true", help="Skip final build_indexes step")
    p.add_argument(
        "--status",
        action="store_true",
        help="Report file counts and cache readiness (no processing)",
    )
    return p.parse_args()


def _count_glob(d: Path, pattern: str) -> int:
    if not d.is_dir():
        return 0
    return len(list(d.glob(pattern)))


def _cache_rows(cache_path: Path, key: str) -> int:
    if not cache_path.is_file():
        return 0
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return len(data.get(key, {}))
    except Exception:
        return 0


def report_status(splits: list[str]) -> int:
    """Print per-split readiness; return 0 if all requested splits are complete."""
    print(f"repo    : {IMAGENET100_REPO_ID}")
    all_ok = True
    for name in splits:
        cfg = SPLIT_CONFIG[name]
        d: Path = cfg["samples_dir"]
        n_b64 = _count_glob(d, "???????.b64")
        n_json = _count_glob(d, "???????.json")
        n_npy = _count_glob(d, "???????.npy")
        n_shapes = _cache_rows(cfg["shape_cache"], "shapes")
        n_labels = _cache_rows(cfg["label_cache"], "labels")
        expected = _EXPECTED_ROWS[name]

        files_ok = n_b64 == n_json == n_npy == expected
        caches_ok = n_shapes == n_labels == expected
        split_ok = files_ok and caches_ok
        all_ok = all_ok and split_ok

        tag = "READY" if split_ok else "INCOMPLETE"
        print(f"\n[{name}] {tag}  (expected {expected:,} rows)")
        print(f"  dir     : {d}")
        print(f"  .b64    : {n_b64:,}")
        print(f"  .json   : {n_json:,}")
        print(f"  .npy    : {n_npy:,}")
        print(f"  shapes  : {cfg['shape_cache'].name}  ({n_shapes:,} rows)")
        print(f"  labels  : {cfg['label_cache'].name}  ({n_labels:,} rows)")
        if not split_ok:
            if n_b64 < expected:
                print(f"  -> run: python scripts/prepare_dataset.py --split {name} --resume")
            elif not caches_ok:
                print(f"  -> run: python scripts/prepare_dataset.py --split {name} --indexes-only")

    print()
    if all_ok:
        print("All splits ready for train_generator_local.py")
        print("  python train_generator_local.py --use-hf-val --epochs 30 --batch-size 4")
    else:
        print("Some splits incomplete — use commands above to finish.")
    return 0 if all_ok else 1


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(MY_WORK))


def main() -> int:
    args = parse_args()

    if args.status:
        splits = ["train", "validation"] if args.split is None else [args.split]
        return report_status(splits)

    if args.split is None:
        print("ERROR: --split train|validation is required (unless --status)", file=sys.stderr)
        return 2

    py = sys.executable

    if not args.indexes_only:
        save_cmd = [
            py, str(SCRIPTS / "save_images.py"),
            "--hf-split", args.split,
            "--workers", str(args.workers),
        ]
        if args.resume:
            save_cmd.append("--resume")
        if args.limit > 0:
            save_cmd.extend(["--limit", str(args.limit)])
        run_step(f"step 1/4: save .b64 ({args.split})", save_cmd)

        infer_cmd = [
            py, str(SCRIPTS / "run_inference.py"),
            "--hf-split", args.split,
            "--batch-size", str(args.batch_size),
            "--workers", str(args.inference_workers),
        ]
        if args.resume:
            infer_cmd.append("--resume")
        run_step(f"step 2/4: inference -> .json ({args.split})", infer_cmd)

        npy_cmd = [
            py, str(SCRIPTS / "save_decoded_npy.py"),
            "--hf-split", args.split,
            "--workers", str(args.workers),
        ]
        if args.resume:
            npy_cmd.append("--resume")
        if args.limit > 0:
            npy_cmd.extend(["--limit", str(args.limit)])
        run_step(f"step 3/4: decode -> .npy ({args.split})", npy_cmd)

    if not args.skip_indexes:
        run_step(
            f"step 4/4: build indexes ({args.split})",
            [py, str(SCRIPTS / "build_indexes.py"), "--split", args.split],
        )

    print(f"\nall done — {args.split} split ready")
    report_status([args.split])
    if args.split == "validation":
        print("\ntrain with HF validation set:")
        print("  python train_generator_local.py --use-hf-val ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
