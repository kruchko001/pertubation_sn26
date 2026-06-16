"""Classifier, loss functions, and byte-accurate pipeline helpers."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from perturb_mirror.constants import MAX_LINF_DELTA, MIN_LINF_DELTA, MIN_PSNR_DB, MIN_SSIM
from perturb_mirror.model import (
    PREPROCESS,
    load_efficientnet_v2_l,
    normalize_prediction_label,
    predict_label,
)


# ─── Classifier ───────────────────────────────────────────────────────────────

def load_frozen_classifier(device: torch.device) -> torch.nn.Module:
    model = load_efficientnet_v2_l(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


# ─── STE quantization ─────────────────────────────────────────────────────────

def quantize_ste(image_bchw: torch.Tensor) -> torch.Tensor:
    """
    uint8 rounding via STE (keeps gradient flowing).
    PNG is lossless, so rounding to 8-bit is the only change.
    """
    scaled = image_bchw * 255.0
    rounded = torch.round(scaled)
    ste = (rounded - scaled).detach() + scaled
    return ste / 255.0


# ─── Validator preprocess ─────────────────────────────────────────────────────

def apply_validator_preprocess(image_bchw: torch.Tensor) -> torch.Tensor:
    """Resize + crop to 480 + ImageNet normalise — same as PREPROCESS."""
    return PREPROCESS(image_bchw)


# ─── True-label helpers ───────────────────────────────────────────────────────

def true_label_indices(model: torch.nn.Module, clean_bchw: torch.Tensor) -> torch.Tensor:
    """Model argmax on clean (no grad) — validator true_label source."""
    with torch.no_grad():
        return model(apply_validator_preprocess(clean_bchw)).argmax(dim=1)


def true_label_strings(model: torch.nn.Module, clean_bchw: torch.Tensor) -> list[str]:
    with torch.no_grad():
        return [
            normalize_prediction_label(predict_label(model, clean_bchw[i]))
            for i in range(int(clean_bchw.shape[0]))
        ]


# ─── Losses ───────────────────────────────────────────────────────────────────

def cw_loss(logits: torch.Tensor, target_indices: torch.Tensor, confidence: float = 0.0) -> torch.Tensor:
    """Untargeted C&W loss — minimise to flip away from true class."""
    idx = target_indices.view(-1, 1)
    target_logits = logits.gather(1, idx).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, idx, False)
    other_logits = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    return torch.clamp(target_logits - other_logits + confidence, min=0.0).mean()


def ssim_loss_differentiable(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    kernel_size: int = 11,
) -> torch.Tensor:
    """
    Differentiable SSIM loss matching perturb_mirror.scoring.compute_ssim exactly.
    Returns (1 - SSIM).mean() — minimise to stay visually close to clean.
    """
    padding = kernel_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x_clean, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(x_adv, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x_clean * x_clean, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(x_adv * x_adv, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x_clean * x_adv, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return (1.0 - ssim_map).mean()


def psnr_loss_differentiable(x_clean: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    """MSE-based surrogate — minimise to maximise PSNR."""
    return torch.mean((x_adv - x_clean) ** 2)


# ─── Full forward pass ────────────────────────────────────────────────────────

def forward_adv(
    model: torch.nn.Module,
    generator: torch.nn.Module,
    clean_bchw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generator → STE quantize → PREPROCESS → logits.
    Pipeline:  clean [0,1] → δ → adv → uint8 STE → PREPROCESS → EfficientNet
    Returns (logits, adv_quant).
    """
    perturbation = generator(clean_bchw)
    adv = torch.clamp(clean_bchw + perturbation, 0.0, 1.0)
    adv_quant = quantize_ste(adv)
    logits = model(apply_validator_preprocess(adv_quant))
    return logits, adv_quant


# ─── Validator-grade evaluation (no grad) ─────────────────────────────────────

def eval_batch(
    model: torch.nn.Module,
    generator: torch.nn.Module,
    clean_bchw: torch.Tensor,
    true_labels: list[str],
    epsilon: float,
    min_linf_delta: float = MIN_LINF_DELTA,
    min_ssim: float = MIN_SSIM,
    min_psnr_db: float = MIN_PSNR_DB,
) -> dict[str, float]:
    """
    Check all validator gates per-image (without base64 encoding overhead).
    Gates: min_delta, max_delta, label_flip, min_ssim, min_psnr_db.
    """
    batch = int(clean_bchw.shape[0])
    if batch == 0:
        return {"pass_rate": 0.0, "flip_rate": 0.0, "ssim_mean": 0.0, "psnr_mean": 0.0}

    with torch.no_grad():
        perturbation = generator(clean_bchw)
        adv = torch.clamp(clean_bchw + perturbation, 0.0, 1.0)
        adv_quant = quantize_ste(adv)

    passed = 0
    flipped = 0
    ssim_sum = 0.0
    psnr_sum = 0.0
    # Mirror verify_and_score: effective ceiling = min(challenge.epsilon, max_linf_delta)
    effective_max_delta = min(epsilon, MAX_LINF_DELTA)

    for i in range(batch):
        c = clean_bchw[i]
        a = adv_quant[i]
        delta = float((a - c).abs().max().item())

        if delta < min_linf_delta:
            continue
        if delta > effective_max_delta:
            continue

        # SSIM (validator formula)
        padding = 5
        ksize = 11
        x = c.unsqueeze(0)
        y = a.unsqueeze(0)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_x = F.avg_pool2d(x, ksize, stride=1, padding=padding)
        mu_y = F.avg_pool2d(y, ksize, stride=1, padding=padding)
        sx = F.avg_pool2d(x * x, ksize, stride=1, padding=padding) - mu_x * mu_x
        sy = F.avg_pool2d(y * y, ksize, stride=1, padding=padding) - mu_y * mu_y
        sxy = F.avg_pool2d(x * y, ksize, stride=1, padding=padding) - mu_x * mu_y
        ssim = float(((2 * mu_x * mu_y + c1) * (2 * sxy + c2) /
                      ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2) + 1e-12)).mean().item())
        ssim_sum += ssim
        if ssim < min_ssim:
            continue

        mse = float(torch.mean((a - c) ** 2).item())
        psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        psnr_sum += psnr
        if min_psnr_db > 0.0 and psnr < min_psnr_db:
            continue

        pred = normalize_prediction_label(predict_label(model, a))
        if pred == true_labels[i]:
            continue

        passed += 1
        flipped += 1

    return {
        "pass_rate": passed / batch,
        "flip_rate": flipped / batch,
        "ssim_mean": ssim_sum / batch,
        "psnr_mean": psnr_sum / batch,
    }
