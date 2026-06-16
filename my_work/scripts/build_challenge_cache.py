#!/usr/bin/env python3
"""
Build a local JSONL cache of (clean_image_b64, true_label) pairs
that exactly match what the validator sends to miners.

Pipeline per row (mirrors neurons/validator.py dev branch):
  HF dataset[row]["image"]
      → convert("RGB")
      → JPEG quality=95 → bytes
      → base64                      ← clean_image_b64
      → decode_image_b64            ← float [0,1] CHW
      → EfficientNet PREPROCESS
      → predict_label → normalize   ← true_label

Output: JSONL, one JSON object per line:
  {"row": int, "image_id": str, "true_label": str, "clean_image_b64": str}

Saved to: my_work/data/imagenet100_samples/challenge_cache.jsonl
(gitignored — local only)

Usage (from my_work/):
  python scripts/build_challenge_cache.py
  python scripts/build_challenge_cache.py --limit 1000
  python scripts/build_challenge_cache.py --limit 0          # full ~126k
  python scripts/build_challenge_cache.py --resume           # skip already-done rows
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from paths import IMAGENET100_SAMPLES_DIR
from perturb_mirror.imagenet100_bootstrap import imagenet100_dataset_version, load_imagenet100
from perturb_mirror.model import load_efficientnet_v2_l, normalize_prediction_label, predict_label

CACHE_FILE = IMAGENET100_SAMPLES_DIR / "challenge_cache.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build validator-matched challenge cache")
    p.add_argument("--limit", type=int, default=0,
                   help="Max rows to process (0 = full dataset)")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already present in the cache file")
    p.add_argument("--log-every", type=int, default=500,
                   help="Print progress every N rows")
    p.add_argument("--output", type=Path, default=CACHE_FILE,
                   help=f"Output JSONL path (default: {CACHE_FILE})")
    return p.parse_args()


def row_to_clean_b64(pil_image) -> str:
    """
    Mirrors _imagenet100_image_bytes():
      PIL Image → convert("RGB") → JPEG quality=95 → bytes → base64
    """
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_tensor(clean_b64: str, device: torch.device) -> torch.Tensor:
    """
    Mirrors decode_image_b64():
      base64 → JPEG bytes → PIL.open → RGB → float32 [0,1] CHW
    """
    from PIL import Image
    raw = base64.b64decode(clean_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load already-cached rows if resuming
    done_rows: set[int] = set()
    if args.resume and args.output.exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_rows.add(int(obj["row"]))
                except Exception:
                    pass
        print(f"resume: {len(done_rows)} rows already cached, skipping them")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")

    print("loading ImageNet-100 dataset …")
    dataset = load_imagenet100()
    version = imagenet100_dataset_version(
        dataset=dataset,
        repo_id="clane9/imagenet-100",
        split="train",
    )
    total_rows = int(dataset.num_rows)
    print(f"dataset: {total_rows} rows  version={version}")

    print("loading EfficientNet-V2-L …")
    model = load_efficientnet_v2_l(device)

    # Determine which rows to process
    rows_to_process = [r for r in range(total_rows) if r not in done_rows]
    if args.limit > 0:
        rows_to_process = rows_to_process[: args.limit]
    print(f"rows to process: {len(rows_to_process)}")

    mode = "a" if args.resume else "w"
    t0 = time.time()
    written = 0
    errors = 0

    with open(args.output, mode, encoding="utf-8") as out:
        for i, row in enumerate(rows_to_process, start=1):
            try:
                pil_img = dataset[row]["image"]
                clean_b64 = row_to_clean_b64(pil_img)
                tensor = b64_to_tensor(clean_b64, device)
                predicted = predict_label(model, tensor)
                true_label = normalize_prediction_label(predicted)
                image_id = f"hf-{version}-{row:07d}"

                record = {
                    "row": row,
                    "image_id": image_id,
                    "true_label": true_label,
                    "clean_image_b64": clean_b64,
                }
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1

            except Exception as exc:
                errors += 1
                print(f"  [row {row}] ERROR: {exc}", file=sys.stderr)
                continue

            if args.log_every > 0 and i % args.log_every == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(rows_to_process) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i:>7}/{len(rows_to_process)}]  "
                    f"written={written}  errors={errors}  "
                    f"rate={rate:.1f} rows/s  ETA={eta/60:.1f} min"
                )

    elapsed = time.time() - t0
    print(f"\ndone  written={written}  errors={errors}  "
          f"elapsed={elapsed/60:.1f} min  output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
