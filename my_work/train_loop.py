"""Shared train/validation loops for generator training scripts."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from generator import Generator
from model_utils import (
    clean_feature_maps,
    cw_loss,
    dlr_loss,
    eval_batch,
    feat_separation_loss,
    flip_first_loss,
    forward_adv,
    forward_adv_with_feats,
    mae_aligned_loss,
    psnr_loss_differentiable,
    reward_aligned_loss,
    ssim_loss_differentiable,
    true_label_indices,
)


def train_one_epoch(
    generator: Generator,
    classifier: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    log_every: int,
    desc: str = "train",
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    quantize: bool = True,
) -> dict[str, float]:
    generator.train()
    if args.loss in ("reward", "feat", "flipfirst"):
        totals: dict[str, float] = {
            "loss": 0.0, "flip": 0.0, "score": 0.0, "floor": 0.0,
            "ssim_h": 0.0, "psnr_h": 0.0, "flip_rate": 0.0,
            "pert_score": 0.0, "linf_mean": 0.0,
        }
        if args.loss == "feat":
            totals["feat_dist"] = 0.0
        if args.loss == "flipfirst":
            totals["robust_rate"] = 0.0
    elif args.loss == "mae":
        totals = {
            "loss": 0.0, "flip": 0.0, "mae": 0.0, "floor": 0.0,
            "flip_rate": 0.0, "pert_score": 0.0, "linf_mean": 0.0,
        }
    else:
        totals = {"loss": 0.0, "cw": 0.0, "ssim": 0.0, "psnr": 0.0}
    steps = 0

    channels_last = (not args.no_channels_last) and device.type == "cuda"
    grad_checkpoint = bool(getattr(args, "grad_checkpoint", False))
    checkpoint_segments = int(getattr(args, "checkpoint_segments", 4))
    feat_layers = tuple(getattr(args, "feat_layers", (4,)))
    feat_normalize = bool(getattr(args, "feat_normalize", True))

    # Flip driver selection. DLR is scale-invariant, so its hinge confidence is
    # on the normalised DLR scale (~O(1)), not the raw-logit scale used by CW.
    flip_loss = str(getattr(args, "flip_loss", "cw"))
    flip_confidence = (
        float(getattr(args, "dlr_confidence", 0.1))
        if flip_loss == "dlr"
        else float(getattr(args, "cw_confidence", 6.0))
    )
    floor_ungated = bool(getattr(args, "floor_ungated", False))

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for step, (clean, labels) in enumerate(pbar, start=1):
        clean = clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if (labels < 0).any():
            computed = true_label_indices(classifier, clean)
            labels = torch.where(labels < 0, computed, labels)
        target_idx = labels

        if args.loss == "feat":
            feats_clean = clean_feature_maps(classifier, clean, feat_layers)
            logits, adv_quant, feats_adv = forward_adv_with_feats(
                classifier, generator, clean, feat_layers,
                channels_last=channels_last,
                grad_checkpoint=grad_checkpoint,
                quantize=quantize,
            )
        else:
            logits, adv_quant = forward_adv(
                classifier, generator, clean,
                channels_last=channels_last,
                grad_checkpoint=grad_checkpoint,
                checkpoint_segments=checkpoint_segments,
                quantize=quantize,
            )

        if args.loss in ("reward", "feat"):
            flip_override = None
            feat_dist_val = 0.0
            if args.loss == "feat":
                # Flip driver = maximise clean<->adv feature distance (negate to
                # minimise). w_feat folds the weight in; reward_aligned_loss then
                # leaves w_flip unapplied for the override term.
                feat_dist = feat_separation_loss(feats_adv, feats_clean, normalize=feat_normalize)
                flip_override = -args.w_feat * feat_dist
                feat_dist_val = float(feat_dist.item())
            loss, comps = reward_aligned_loss(
                logits=logits,
                target_indices=target_idx,
                clean_bchw=clean,
                adv_quant=adv_quant,
                cw_confidence=flip_confidence,
                floor_margin=args.floor_margin,
                linf_topk=args.linf_topk,
                w_flip=args.w_flip,
                w_score=args.w_score,
                w_floor=args.w_floor,
                w_ssim=args.w_ssim,
                w_psnr=args.w_psnr,
                flip_loss=flip_loss,
                flip_loss_override=flip_override,
                floor_ungated=floor_ungated,
            )
            if args.loss == "feat":
                comps["feat_dist"] = feat_dist_val
        elif args.loss == "flipfirst":
            loss, comps = flip_first_loss(
                logits=logits,
                target_indices=target_idx,
                clean_bchw=clean,
                adv_quant=adv_quant,
                cw_confidence=flip_confidence,
                robust_margin=getattr(args, "robust_margin", None),
                floor_margin=args.floor_margin,
                linf_topk=args.linf_topk,
                w_flip=args.w_flip,
                w_score=args.w_score,
                w_floor=args.w_floor,
                w_ssim=args.w_ssim,
                w_psnr=args.w_psnr,
                flip_loss=flip_loss,
                floor_ungated=floor_ungated,
            )
        elif args.loss == "mae":
            loss, comps = mae_aligned_loss(
                logits=logits,
                target_indices=target_idx,
                clean_bchw=clean,
                adv_quant=adv_quant,
                cw_confidence=flip_confidence,
                floor_margin=args.floor_margin,
                w_flip=args.w_flip,
                w_mae=args.w_mae,
                w_floor=args.w_floor,
                flip_loss=flip_loss,
            )
        else:
            flip_fn = dlr_loss if flip_loss == "dlr" else cw_loss
            loss_cw = flip_fn(logits, target_idx)
            loss_ssim = ssim_loss_differentiable(clean, adv_quant)
            loss_psnr = psnr_loss_differentiable(clean, adv_quant)
            loss = loss_cw + args.ssim_weight * loss_ssim + args.psnr_weight * loss_psnr
            comps = {
                "loss": float(loss.item()), "cw": float(loss_cw.item()),
                "ssim": float(loss_ssim.item()), "psnr": float(loss_psnr.item()),
            }

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k in totals:
            totals[k] += comps.get(k, 0.0)
        steps += 1

        if log_every > 0 and step % log_every == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            if args.loss in ("reward", "feat", "flipfirst"):
                postfix = {
                    "loss": f"{comps['loss']:.3f}",
                    "flip": f"{comps['flip']:.3f}",
                    "score": f"{comps['score']:.3f}",
                    "floor": f"{comps['floor']:.3f}",
                    "ssim_h": f"{comps['ssim_h']:.3f}",
                    "psnr_h": f"{comps['psnr_h']:.3f}",
                    "fr": f"{comps['flip_rate']:.3f}",
                    "ps": f"{comps['pert_score']:.3f}",
                    "linf": f"{comps['linf_mean']:.5f}",
                    "lr": f"{cur_lr:.2e}",
                }
                if args.loss == "feat":
                    postfix["fdist"] = f"{comps['feat_dist']:.3f}"
                if args.loss == "flipfirst":
                    postfix["rr"] = f"{comps['robust_rate']:.3f}"
                pbar.set_postfix(postfix, refresh=False)
            elif args.loss == "mae":
                pbar.set_postfix(
                    {
                        "loss": f"{comps['loss']:.3f}",
                        "flip": f"{comps['flip']:.3f}",
                        "mae": f"{comps['mae']:.5f}",
                        "floor": f"{comps['floor']:.3f}",
                        "fr": f"{comps['flip_rate']:.3f}",
                        "ps": f"{comps['pert_score']:.3f}",
                        "linf": f"{comps['linf_mean']:.5f}",
                        "lr": f"{cur_lr:.2e}",
                    },
                    refresh=False,
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{comps['loss']:.3f}",
                        "cw": f"{comps['cw']:.3f}",
                        "ssim": f"{comps['ssim']:.5f}",
                        "psnr": f"{comps['psnr']:.6f}",
                        "lr": f"{cur_lr:.2e}",
                    },
                    refresh=False,
                )

    pbar.close()
    denom = max(steps, 1)
    return {k: v / denom for k, v in totals.items()}


@torch.no_grad()
def validate(
    generator: Generator,
    classifier: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float,
) -> dict[str, float]:
    generator.eval()
    agg: dict[str, float] = {
        "pass_rate": 0.0, "flip_rate": 0.0, "ssim_mean": 0.0,
        "psnr_mean": 0.0, "score_mean": 0.0,
    }
    steps = 0

    pbar = tqdm(loader, desc="val", dynamic_ncols=True, leave=True)
    for clean, labels in pbar:
        clean = clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if (labels < 0).any():
            computed = true_label_indices(classifier, clean)
            labels = torch.where(labels < 0, computed, labels)
        stats = eval_batch(
            model=classifier,
            generator=generator,
            clean_bchw=clean,
            true_indices=labels,
            epsilon=epsilon,
        )
        for k in agg:
            agg[k] += stats[k]
        steps += 1
        pbar.set_postfix(
            {
                "score": f"{agg['score_mean'] / steps:.4f}",
                "pass": f"{agg['pass_rate'] / steps:.3f}",
                "fr": f"{agg['flip_rate'] / steps:.3f}",
            },
            refresh=False,
        )
    pbar.close()

    denom = max(steps, 1)
    return {k: v / denom for k, v in agg.items()}
