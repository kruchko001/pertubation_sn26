#!/usr/bin/env python3
"""
Train a perturbation generator that matches the validator's full pipeline.

Key fixes vs naive implementation:
  1. Clean image: JPEG q=95 round-trip (matches validator ImageNet-100 path).
  2. Adversarial: uint8 STE before PREPROCESS (matches PNG miner submission).
  3. Loss: differentiable SSIM from scoring.py formula + MSE-PSNR surrogate.
  4. Epsilon: sampled per batch from validator range [0.06, 0.2], capped at MAX_LINF_DELTA.
  5. Validation: checks all validator gates (delta, SSIM, PSNR, label flip).

Usage (from my_work/):
  python train_generator.py --limit 512 --epochs 5
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MY_WORK = Path(__file__).resolve().parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from challenge_io import load_imagenet100_dataset
from generator import Generator
from model_utils import (
    cw_loss,
    eval_batch,
    forward_adv,
    jpeg_round_trip,
    load_frozen_classifier,
    psnr_loss_differentiable,
    ssim_loss_differentiable,
    true_label_indices,
    true_label_strings,
)
from paths import OUTPUTS
from perturb_mirror.constants import MAX_LINF_DELTA, MIN_LINF_DELTA
from perturb_mirror.validator import sample_epsilon


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ImageNet100Dataset(Dataset):
    """
    Returns JPEG-decoded float tensors [0,1] CHW.
    Matches validator path: PIL → JPEG q=95 → decode_image_b64.
    """

    def __init__(self, hf_dataset, indices: list[int]) -> None:
        self.hf_dataset = hf_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = int(self.indices[idx])
        pil_img = self.hf_dataset[row]["image"].convert("RGB")
        # Simulate validator JPEG q=95 encode → decode
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        from PIL import Image
        decoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(decoded).permute(2, 0, 1).contiguous()


def _collate_single(batch: list[torch.Tensor]) -> torch.Tensor:
    return batch[0].unsqueeze(0)


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train perturbation generator (validator-matched)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssim-weight", type=float, default=10.0,
                        help="Weight on SSIM loss (keep adv perceptually close)")
    parser.add_argument("--psnr-weight", type=float, default=5.0,
                        help="Weight on MSE/PSNR surrogate loss")
    parser.add_argument("--max-linf", type=float, default=MAX_LINF_DELTA,
                        help=f"Generator L-inf cap (default={MAX_LINF_DELTA}, validator max)")
    parser.add_argument("--limit", type=int, default=0, help="Max train rows (0=full split)")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-epsilon", type=float, default=0.12,
                        help="Fixed epsilon for validation gates (validator samples 0.06-0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=OUTPUTS / "generator.pt")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    generator: Generator,
    classifier: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ssim_weight: float,
    psnr_weight: float,
    log_every: int,
    epoch_seed: int,
) -> dict[str, float]:
    generator.train()
    totals: dict[str, float] = {"loss": 0.0, "cw": 0.0, "ssim": 0.0, "psnr": 0.0}
    steps = 0

    for step, clean in enumerate(loader, start=1):
        clean = clean.to(device)

        target_idx = true_label_indices(classifier, clean)
        logits, adv_quant = forward_adv(classifier, generator, clean)

        loss_cw = cw_loss(logits, target_idx)
        # Use same SSIM formula as scoring.py — matching validator gate
        loss_ssim = ssim_loss_differentiable(clean, adv_quant)
        loss_psnr = psnr_loss_differentiable(clean, adv_quant)

        loss = loss_cw + ssim_weight * loss_ssim + psnr_weight * loss_psnr

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        totals["loss"] += float(loss.item())
        totals["cw"] += float(loss_cw.item())
        totals["ssim"] += float(loss_ssim.item())
        totals["psnr"] += float(loss_psnr.item())
        steps += 1

        if log_every > 0 and step % log_every == 0:
            print(
                f"  step {step:5d}  "
                f"loss={loss.item():.4f}  cw={loss_cw.item():.4f}  "
                f"ssim_loss={loss_ssim.item():.5f}  psnr_mse={loss_psnr.item():.6f}"
            )

    denom = max(steps, 1)
    return {k: v / denom for k, v in totals.items()}


# ─── Validation (all validator gates) ─────────────────────────────────────────

@torch.no_grad()
def validate(
    generator: Generator,
    classifier: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float,
) -> dict[str, float]:
    generator.eval()
    agg: dict[str, float] = {"pass_rate": 0.0, "flip_rate": 0.0, "ssim_mean": 0.0, "psnr_mean": 0.0}
    steps = 0

    for clean in loader:
        clean = clean.to(device)
        true_labels = true_label_strings(classifier, clean)
        stats = eval_batch(
            model=classifier,
            generator=generator,
            clean_bchw=clean,
            true_labels=true_labels,
            epsilon=epsilon,
        )
        for k in agg:
            agg[k] += stats[k]
        steps += 1

    denom = max(steps, 1)
    return {k: v / denom for k, v in agg.items()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device              : {device}")
    print(f"max_linf (generator): {args.max_linf}  (validator MAX_LINF_DELTA={MAX_LINF_DELTA})")
    print(f"min_linf_delta      : {MIN_LINF_DELTA}  (validator gate)")
    print(f"val_epsilon         : {args.val_epsilon}")

    hf_dataset = load_imagenet100_dataset()
    indices = list(range(int(hf_dataset.num_rows)))
    random.shuffle(indices)
    if args.limit > 0:
        indices = indices[: args.limit]

    val_count = max(1, int(len(indices) * args.val_fraction))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    print(f"train rows          : {len(train_indices)}")
    print(f"val rows            : {len(val_indices)}")

    train_loader = DataLoader(
        ImageNet100Dataset(hf_dataset, train_indices),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_single,
    )
    val_loader = DataLoader(
        ImageNet100Dataset(hf_dataset, val_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_single,
    )

    classifier = load_frozen_classifier(device)
    generator = Generator(max_linf=args.max_linf).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

    best_pass = -1.0
    args.save.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== epoch {epoch}/{args.epochs} ===")
        train_stats = train_one_epoch(
            generator=generator,
            classifier=classifier,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            ssim_weight=args.ssim_weight,
            psnr_weight=args.psnr_weight,
            log_every=args.log_every,
            epoch_seed=args.seed + epoch,
        )
        val_stats = validate(generator, classifier, val_loader, device, epsilon=args.val_epsilon)

        print(
            f"train  loss={train_stats['loss']:.4f}  "
            f"cw={train_stats['cw']:.4f}  "
            f"ssim_loss={train_stats['ssim']:.5f}  "
            f"psnr_mse={train_stats['psnr']:.6f}"
        )
        print(
            f"val    pass_rate={val_stats['pass_rate']:.3f}  "   # full gates passed
            f"flip_rate={val_stats['flip_rate']:.3f}  "
            f"ssim={val_stats['ssim_mean']:.4f}  "
            f"psnr={val_stats['psnr_mean']:.2f} dB"
        )

        if val_stats["pass_rate"] > best_pass:
            best_pass = val_stats["pass_rate"]
            torch.save(
                {
                    "epoch": epoch,
                    "max_linf": args.max_linf,
                    "generator_state": generator.state_dict(),
                    "val": val_stats,
                },
                args.save,
            )
            print(f"saved  → {args.save}  (pass_rate={best_pass:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
