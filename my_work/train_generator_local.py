#!/usr/bin/env python3
"""
Train the perturbation generator from local .npy files only (no Hugging Face).

Requires pre-generated assets:

  Train (HF train split, 126689 rows):
    data/imagenet100_samples/{row:07d}.npy
    data/imagenet100_shapes.json
    data/imagenet100_true_labels.json

  Validation (HF validation split, 5000 rows) — optional but recommended:
    data/imagenet100_val_samples/{row:07d}.npy
    data/imagenet100_val_shapes.json
    data/imagenet100_val_true_labels.json

Prepare validation data (same pipeline as train):
  python scripts/prepare_dataset.py --split validation --workers 8 --resume

Rebuild train indexes after adding files:
  python scripts/build_indexes.py --split train

Usage (from my_work/):
  # default: train-only, NO per-epoch validation, best checkpoint by train score
  python train_generator_local.py --epochs 30 --batch-size 4 --workers 8

  # opt back into per-epoch validation on the HF val split:
  python train_generator_local.py --val --use-hf-val --epochs 30 --batch-size 4

  # LTP-style: feature-separation flip driver (keeps reward L-inf/SSIM/PSNR terms),
  # tap a mid-level EfficientNetV2-L block; sweep --feat-layers / --w-feat to tune:
  python train_generator_local.py --loss feat --feat-layers 4 --w-feat 10 --epochs 30

  # LTP ResNet generator instead of the U-Net (LTP uses base=64):
  python train_generator_local.py --gen-arch resnet --gen-base 64 --loss feat --feat-layers 4

  # margin-CW flip + MAE perturbation restriction (L-inf<=1/255 from generator,
  # min-L-inf floor protects the 0.003 gate); tune --w-mae to trade flip vs size:
  python train_generator_local.py --loss mae --w-mae 100 --epochs 30 --batch-size 4
"""

from __future__ import annotations

import argparse
import math
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
from generator import build_generator, load_generator_checkpoint
from local_index import (
    NpyDataset,
    discover_npy_rows,
    load_label_index,
    load_shape_index,
)
from model_utils import load_frozen_classifier
from paths import (
    IMAGENET100_LABELS_CACHE,
    IMAGENET100_SAMPLES_DIR,
    IMAGENET100_SHAPES_CACHE,
    IMAGENET100_VAL_LABELS_CACHE,
    IMAGENET100_VAL_SAMPLES_DIR,
    IMAGENET100_VAL_SHAPES_CACHE,
    OUTPUTS,
)
from perturb_mirror.constants import MAX_LINF_DELTA, MIN_LINF_DELTA
from train_loop import train_one_epoch, validate

torch.set_float32_matmul_precision("high")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train generator from local .npy + index caches (no Hugging Face)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=["cosine", "none"], default="cosine",
                        help="LR schedule: cosine (default)=linear warmup then cosine "
                             "decay to --min-lr over all steps; none=constant lr.")
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="Linear LR warmup steps (0..--lr) before cosine decay. "
                             "Smooths the early, high-variance steps so the generator "
                             "isn't kicked around before it stabilises.")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Cosine decay floor (final lr at the end of training).")
    parser.add_argument("--loss", choices=["reward", "legacy", "feat", "mae"], default="reward",
                        help="reward=margin-CW flip driver; feat=LTP mid-level "
                             "feature-separation flip driver (keeps reward L-inf/SSIM/PSNR "
                             "terms); mae=margin-CW flip driver + MAE perturbation restriction "
                             "+ min-L-inf floor (L-inf<=1/255 enforced by the generator); "
                             "legacy=plain CW+SSIM+PSNR")
    parser.add_argument("--flip-loss", choices=["cw", "dlr"], default="dlr",
                        help="Flip driver for reward/feat/mae/legacy losses. "
                             "dlr (default)=Difference-of-Logits-Ratio: normalises the "
                             "logit margin by the logit spread, so it is invariant to "
                             "per-image logit scale and a single confident image can't "
                             "blow up the batch loss (stable training). cw=raw-logit "
                             "margin (legacy; unbounded scale).")
    parser.add_argument("--dlr-confidence", type=float, default=0.1,
                        help="With --flip-loss dlr: hinge margin on the NORMALISED DLR "
                             "scale (~O(1)), not raw logits. The flip term switches off "
                             "once an image is flipped by this margin.")
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
    parser.add_argument("--w-mae", type=float, default=100.0,
                        help="With --loss mae: weight on the MAE (mean |perturbation|) "
                             "restriction term (flip-gated). MAE is on the ~[0,1/255] scale "
                             "so this needs to be large to balance the margin-CW flip term.")
    parser.add_argument("--max-linf", type=float, default=MAX_LINF_DELTA)
    parser.add_argument("--gen-arch", choices=["unet", "resnet"], default="unet",
                        help="Generator architecture: unet (default, resolution-agnostic) "
                             "or resnet (LTP GeneratorResnet port)")
    parser.add_argument("--gen-base", type=int, default=48,
                        help="Generator base channel width (capacity; e.g. 32/48/64). "
                             "LTP uses 64 for the resnet arch.")
    parser.add_argument("--gen-dropout", type=float, default=0.0,
                        help="Dropout in resnet generator residual blocks (LTP used 0.5)")
    parser.add_argument("--feat-layers", type=str, default="4",
                        help="With --loss feat: comma-separated EfficientNetV2-L feature "
                             "block indices to separate (0..8; mid-level ~3-5)")
    parser.add_argument("--w-feat", type=float, default=10.0,
                        help="With --loss feat: weight on the (normalised) feature-separation "
                             "flip driver")
    parser.add_argument("--feat-normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalise each layer's feature distance by clean feature energy "
                             "(stable w_feat across layers; default on)")
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all .npy)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--samples-dir", type=Path, default=IMAGENET100_SAMPLES_DIR)
    parser.add_argument("--shape-cache", type=Path, default=IMAGENET100_SHAPES_CACHE)
    parser.add_argument("--label-cache", type=Path, default=IMAGENET100_LABELS_CACHE)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=False,
                        help="Run per-epoch validation (default OFF; the dataset is train-only). "
                             "Use --val to enable, --no-val to keep it off.")
    parser.add_argument("--use-hf-val", action="store_true",
                        help="With --val: validate on HF validation split (data/imagenet100_val_samples)")
    parser.add_argument("--val-samples-dir", type=Path, default=IMAGENET100_VAL_SAMPLES_DIR)
    parser.add_argument("--val-shape-cache", type=Path, default=IMAGENET100_VAL_SHAPES_CACHE)
    parser.add_argument("--val-label-cache", type=Path, default=IMAGENET100_VAL_LABELS_CACHE)
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Hold out from train npy when --use-hf-val is off")
    parser.add_argument("--val-epsilon", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=OUTPUTS / "generator.pt")
    parser.add_argument("--load", type=Path, default=None,
                        help="Pre-trained checkpoint to fine-tune (generator_state from a prior run)")
    parser.add_argument("--resume", action="store_true",
                        help="With --load: continue best-score tracking from checkpoint val score")
    parser.add_argument("--log-every", type=int, default=3)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Gradient-checkpoint the classifier (saves VRAM, ~20-30%% slower)")
    parser.add_argument("--checkpoint-segments", type=int, default=4,
                        help="Number of checkpoint_sequential segments over model.features")
    args = parser.parse_args()
    args.feat_layers = tuple(
        int(x) for x in str(args.feat_layers).replace(" ", "").split(",") if x != ""
    )
    if args.loss == "feat" and not args.feat_layers:
        parser.error("--loss feat requires at least one --feat-layers index")
    return args


def print_train_summary(args: argparse.Namespace, train_stats: dict[str, float]) -> None:
    if args.loss in ("reward", "feat"):
        fdist = f" fdist={train_stats['feat_dist']:.4f}" if "feat_dist" in train_stats else ""
        print(
            f"train  loss={train_stats['loss']:.4f} | "
            f"flip={train_stats['flip']:.4f} score={train_stats['score']:.4f} "
            f"floor={train_stats['floor']:.4f} ssim_h={train_stats['ssim_h']:.4f} "
            f"psnr_h={train_stats['psnr_h']:.4f}{fdist} | "
            f"flip_rate={train_stats['flip_rate']:.3f} "
            f"pert_score={train_stats['pert_score']:.4f} "
            f"linf_mean={train_stats['linf_mean']:.5f}"
        )
    elif args.loss == "mae":
        print(
            f"train  loss={train_stats['loss']:.4f} | "
            f"flip={train_stats['flip']:.4f} mae={train_stats['mae']:.5f} "
            f"floor={train_stats['floor']:.4f} | "
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


def compute_train_score(args: argparse.Namespace, train_stats: dict[str, float]) -> float:
    """Scalar checkpoint-selection metric from TRAIN stats (higher is better).

    With validation off we still need to pick the best epoch. For the reward
    loss we use a validator-aligned proxy: only flipped images earn their
    perturbation score, so ``flip_rate * pert_score`` tracks the on-chain reward
    without a held-out split. For the legacy loss (no score notion) we fall back
    to negative loss.
    """
    if args.loss in ("reward", "feat", "mae"):
        return float(train_stats.get("flip_rate", 0.0) * train_stats.get("pert_score", 0.0))
    return -float(train_stats.get("loss", float("inf")))


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
    print(f"train npy files     : {len(all_rows)}")

    train_indices = list(all_rows)
    random.shuffle(train_indices)
    if args.limit > 0:
        train_indices = train_indices[: args.limit]

    # Validation is OFF by default: the dataset is train-only, so we train on
    # ALL train rows and select the checkpoint by train score. Pass --val to
    # re-enable per-epoch validation (HF val split with --use-hf-val, else a
    # random holdout carved from the train rows).
    val_indices: list[int] = []
    val_samples_dir = args.val_samples_dir
    val_shape_cache = args.val_shape_cache
    val_label_cache = args.val_label_cache
    if args.val:
        if args.use_hf_val:
            try:
                val_indices = discover_npy_rows(args.val_samples_dir)
            except FileNotFoundError:
                print(f"ERROR: --val --use-hf-val but no .npy in {args.val_samples_dir}")
                print("Run: python scripts/prepare_dataset.py --split validation --workers 8")
                return 1
            if not val_indices:
                print(f"ERROR: --val --use-hf-val but {args.val_samples_dir} is empty")
                return 1
            print(f"val source          : HF validation split")
            print(f"val npy files       : {len(val_indices)}")
            print(f"val samples_dir     : {val_samples_dir}")
        else:
            val_count = max(1, int(len(train_indices) * args.val_fraction))
            val_indices, train_indices = train_indices[:val_count], train_indices[val_count:]
            val_samples_dir = args.samples_dir
            val_shape_cache = args.shape_cache
            val_label_cache = args.label_cache
            print(f"val source          : random holdout ({args.val_fraction:.0%} of train npy)")
    else:
        print(f"val source          : disabled (--no-val) -> best checkpoint by train score")

    print(f"train rows          : {len(train_indices)}")
    print(f"val rows            : {len(val_indices)}")

    # max_pixels = args.max_pixels if args.max_pixels > 0 else args.batch_size * 240 * 240
    max_pixels = args.max_pixels if args.max_pixels > 0 else None
    print(f"batch_size          : {args.batch_size}")
    print(f"max_pixels (cap)    : {max_pixels}")

    used_train_rows = train_indices
    shape_index = load_shape_index(args.shape_cache, used_train_rows)
    labels_map = load_label_index(args.label_cache, used_train_rows)
    print(f"train shape cache   : {len(shape_index)}/{len(used_train_rows)} rows")
    print(f"train label cache   : {len(labels_map)}/{len(used_train_rows)} rows")

    train_shapes = [shape_index[r] for r in train_indices]
    train_sampler = BucketBatchSampler(
        train_shapes,
        batch_size=args.batch_size,
        max_pixels=None,
        shuffle=True,
        drop_last=args.drop_last,
        seed=args.seed,
    )
    print(f"train buckets       : {train_sampler.describe()}")

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

    val_loader = None
    if args.val:
        val_shape_index = load_shape_index(val_shape_cache, val_indices)
        val_labels_map = load_label_index(val_label_cache, val_indices)
        print(f"val shape cache     : {len(val_shape_index)}/{len(val_indices)} rows")
        print(f"val label cache     : {len(val_labels_map)}/{len(val_indices)} rows")
        val_shapes = [val_shape_index[r] for r in val_indices]
        val_sampler = BucketBatchSampler(
            val_shapes,
            batch_size=args.batch_size,
            max_pixels=None,
            shuffle=False,
            drop_last=False,
            seed=args.seed,
        )
        print(f"val buckets         : {val_sampler.describe()}")
        val_loader = DataLoader(
            NpyDataset(val_indices, val_samples_dir, val_labels_map),
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

    generator = build_generator(
        args.gen_arch,
        max_linf=args.max_linf,
        base=args.gen_base,
        gen_dropout=args.gen_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in generator.parameters())
    print(f"generator           : arch={args.gen_arch}  base={args.gen_base}  "
          f"params={n_params/1e6:.2f}M")
    if args.loss == "feat":
        print(f"feat flip driver    : layers={args.feat_layers}  w_feat={args.w_feat}  "
              f"normalize={args.feat_normalize}")

    best_score = -1.0
    if args.load is not None:
        ckpt = load_generator_checkpoint(generator, args.load)
        print(f"loaded checkpoint   : {ckpt['path']}")
        if "epoch" in ckpt:
            print(f"  saved epoch       : {ckpt['epoch']}")
        if "max_linf" in ckpt:
            print(f"  saved max_linf    : {ckpt['max_linf']}")
        if "val" in ckpt and isinstance(ckpt["val"], dict):
            v = ckpt["val"]
            print(
                f"  saved val         : score={v.get('score_mean', 0):.4f}  "
                f"pass={v.get('pass_rate', 0):.3f}  flip={v.get('flip_rate', 0):.3f}"
            )
        if "train_score" in ckpt:
            print(f"  saved train score : {float(ckpt['train_score']):.4f}")
        if args.resume:
            # Resume the best-tracking metric that matches the CURRENT mode so we
            # don't compare a val score against a train score.
            if args.val and isinstance(ckpt.get("val"), dict):
                best_score = float(ckpt["val"].get("score_mean", -1.0))
            elif (not args.val) and ("train_score" in ckpt):
                best_score = float(ckpt["train_score"])
            print(f"  resume best_score : {best_score:.4f}")

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

    scheduler = None
    if args.lr_schedule == "cosine":
        steps_per_epoch = max(1, len(train_loader))
        total_steps = max(1, steps_per_epoch * args.epochs)
        warmup_steps = max(0, min(int(args.warmup_steps), total_steps - 1))
        min_factor = (args.min_lr / args.lr) if args.lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            # step is 0-based count of completed optimizer.step() calls.
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"lr schedule         : cosine (warmup={warmup_steps} steps, "
              f"total={total_steps} steps, min_lr={args.min_lr:g})")
    else:
        print(f"lr schedule         : constant (lr={args.lr:g})")
    print(f"flip loss           : {args.flip_loss}"
          + (f" (dlr_confidence={args.dlr_confidence:g})" if args.flip_loss == "dlr"
             else f" (cw_confidence={args.cw_confidence:g})"))

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
            scheduler=scheduler,
        )
        print_train_summary(args, train_stats)

        if val_loader is not None:
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
                        "gen_arch": args.gen_arch,
                        "generator_state": generator.state_dict(),
                        "val": val_stats,
                    },
                    args.save,
                )
                print(
                    f"saved  -> {args.save}  "
                    f"[constraint max_linf={args.max_linf:.5f} (~{args.max_linf * 255:.2f}/255)]  "
                    f"epoch={epoch}  val score={best_score:.4f}  "
                    f"flip_rate={val_stats['flip_rate']:.3f}"
                )
        else:
            # No validation: select the best epoch by the train-score proxy.
            train_score = compute_train_score(args, train_stats)
            if train_score > best_score:
                best_score = train_score
                torch.save(
                    {
                        "epoch": epoch,
                        "max_linf": args.max_linf,
                        "gen_arch": args.gen_arch,
                        "generator_state": generator.state_dict(),
                        "train": train_stats,
                        "train_score": train_score,
                    },
                    args.save,
                )
                print(
                    f"saved  -> {args.save}  "
                    f"[constraint max_linf={args.max_linf:.5f} (~{args.max_linf * 255:.2f}/255)]  "
                    f"epoch={epoch}  train score={best_score:.4f}  "
                    f"flip_rate={train_stats.get('flip_rate', 0.0):.3f}  "
                    f"linf_mean={train_stats.get('linf_mean', 0.0):.5f}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
