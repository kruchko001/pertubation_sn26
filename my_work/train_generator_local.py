#!/usr/bin/env python3
"""
Train the perturbation generator from local .npy files only (no Hugging Face).

Requires pre-generated assets under data/imagenet100_samples/:
  {row:07d}.npy                    float32 CHW [0,1]  (save_decoded_npy.py)
  data/imagenet100_shapes.json     row -> [H, W]      (train_generator.py once)
  data/imagenet100_true_labels.json row -> label idx  (train_generator.py once)

Usage (from my_work/):
  python train_generator_local.py --epochs 30 --batch-size 24 --workers 8
  python train_generator_local.py --limit 1000 --epochs 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

MY_WORK = Path(__file__).resolve().parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from bucketing import BucketBatchSampler, bucket_collate
from generator import Generator
from local_index import (
    NpyDataset,
    cache_version,
    discover_npy_rows,
    load_label_index,
    load_shape_index,
)
from model_utils import load_frozen_classifier
from paths import DATA, IMAGENET100_SAMPLES_DIR, OUTPUTS
from perturb_mirror.constants import MAX_LINF_DELTA, MIN_LINF_DELTA
from train_loop import train_one_epoch, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train generator from local .npy + index caches (no Hugging Face)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["reward", "legacy"], default="reward")
    parser.add_argument("--ssim-weight", type=float, default=10.0)
    parser.add_argument("--psnr-weight", type=float, default=5.0)
    parser.add_argument("--cw-confidence", type=float, default=6.0)
    parser.add_argument("--floor-margin", type=float, default=0.0005)
    parser.add_argument("--linf-topk", type=int, default=32)
    parser.add_argument("--w-flip", type=float, default=1.0)
    parser.add_argument("--w-score", type=float, default=4.0)
    parser.add_argument("--w-floor", type=float, default=80.0)
    parser.add_argument("--w-ssim", type=float, default=60.0)
    parser.add_argument("--w-psnr", type=float, default=0.05)
    parser.add_argument("--max-linf", type=float, default=MAX_LINF_DELTA)
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all .npy)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--samples-dir", type=Path, default=IMAGENET100_SAMPLES_DIR)
    parser.add_argument("--shape-cache", type=Path, default=DATA / "imagenet100_shapes.json")
    parser.add_argument("--label-cache", type=Path, default=DATA / "imagenet100_true_labels.json")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-epsilon", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=OUTPUTS / "generator.pt")
    parser.add_argument("--log-every", type=int, default=3)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def print_train_summary(args: argparse.Namespace, train_stats: dict[str, float]) -> None:
    if args.loss == "reward":
        print(
            f"train  loss={train_stats['loss']:.4f} | "
            f"flip={train_stats['flip']:.4f} score={train_stats['score']:.4f} "
            f"floor={train_stats['floor']:.4f} ssim_h={train_stats['ssim_h']:.4f} "
            f"psnr_h={train_stats['psnr_h']:.4f} | "
            f"flip_rate={train_stats['flip_rate']:.3f} "
            f"pert_score={train_stats['pert_score']:.4f} "
            f"linf_mean={train_stats['linf_mean']:.5f}"
        )
    else:
        print(
            f"train  loss={train_stats['loss']:.4f}  "
            f"cw={train_stats['cw']:.4f}  "
            f"ssim_loss={train_stats['ssim']:.5f}  "
            f"psnr_mse={train_stats['psnr']:.6f}"
        )


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"device              : {device}")
    print(f"data source         : local .npy only (no Hugging Face)")
    print(f"samples_dir         : {args.samples_dir}")
    print(f"max_linf (generator): {args.max_linf}")
    print(f"min_linf_delta      : {MIN_LINF_DELTA}")
    print(f"val_epsilon         : {args.val_epsilon}")

    all_rows = discover_npy_rows(args.samples_dir)
    print(f"npy files           : {len(all_rows)}")
    shape_ver = cache_version(args.shape_cache)
    label_ver = cache_version(args.label_cache)
    if shape_ver and label_ver and shape_ver != label_ver:
        print(f"WARNING: shape cache version {shape_ver!r} != label cache {label_ver!r}")

    indices = list(all_rows)
    random.shuffle(indices)
    if args.limit > 0:
        indices = indices[: args.limit]

    val_count = max(1, int(len(indices) * args.val_fraction))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    print(f"train rows          : {len(train_indices)}")
    print(f"val rows            : {len(val_indices)}")

    max_pixels = args.max_pixels if args.max_pixels > 0 else args.batch_size * 240 * 240
    print(f"batch_size          : {args.batch_size}")
    print(f"max_pixels (cap)    : {max_pixels}")

    used_rows = train_indices + val_indices
    shape_index = load_shape_index(args.shape_cache, used_rows)
    labels_map = load_label_index(args.label_cache, used_rows)
    print(f"shape cache         : {len(shape_index)}/{len(used_rows)} rows")
    print(f"label cache         : {len(labels_map)}/{len(used_rows)} rows")

    train_shapes = [shape_index[r] for r in train_indices]
    val_shapes = [shape_index[r] for r in val_indices]

    train_sampler = BucketBatchSampler(
        train_shapes,
        batch_size=args.batch_size,
        max_pixels=max_pixels,
        shuffle=True,
        drop_last=args.drop_last,
        seed=args.seed,
    )
    val_sampler = BucketBatchSampler(
        val_shapes,
        batch_size=args.batch_size,
        max_pixels=max_pixels,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
    )
    print(f"train buckets       : {train_sampler.describe()}")
    print(f"val buckets         : {val_sampler.describe()}")

    loader_kwargs = dict(
        num_workers=args.workers,
        collate_fn=bucket_collate,
        pin_memory=(device.type == "cuda"),
    )
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)

    train_loader = DataLoader(
        NpyDataset(train_indices, args.samples_dir, labels_map),
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        NpyDataset(val_indices, args.samples_dir, labels_map),
        batch_sampler=val_sampler,
        **loader_kwargs,
    )

    classifier = load_frozen_classifier(device)
    if device.type == "cuda" and not args.no_channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
        print("channels_last       : enabled (classifier)")
    if args.compile:
        try:
            classifier = torch.compile(classifier)
            print("torch.compile       : enabled (classifier)")
        except Exception as exc:
            print(f"torch.compile       : unavailable ({exc}) -> running eager")
    print(f"dtype               : float32")
    print(f"workers/prefetch    : {args.workers}/{args.prefetch_factor}")

    generator = Generator(max_linf=args.max_linf).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

    best_score = -1.0
    args.save.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            generator=generator,
            classifier=classifier,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            log_every=args.log_every,
            desc=f"epoch {epoch}/{args.epochs}",
        )
        print_train_summary(args, train_stats)

        run_val = (epoch % max(1, args.val_every) == 0) or (epoch == args.epochs)
        if not run_val:
            continue

        val_stats = validate(generator, classifier, val_loader, device, epsilon=args.val_epsilon)
        print(
            f"val    score={val_stats['score_mean']:.4f}  "
            f"pass_rate={val_stats['pass_rate']:.3f}  "
            f"flip_rate={val_stats['flip_rate']:.3f}  "
            f"ssim={val_stats['ssim_mean']:.4f}  "
            f"psnr={val_stats['psnr_mean']:.2f} dB"
        )

        if val_stats["score_mean"] > best_score:
            best_score = val_stats["score_mean"]
            torch.save(
                {
                    "epoch": epoch,
                    "max_linf": args.max_linf,
                    "generator_state": generator.state_dict(),
                    "val": val_stats,
                },
                args.save,
            )
            print(f"saved  -> {args.save}  (score={best_score:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
