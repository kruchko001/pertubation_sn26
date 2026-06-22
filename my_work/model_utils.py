"""Classifier, loss functions, and byte-accurate pipeline helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from perturb_mirror.constants import (
    LINF_COMPONENT_WEIGHT,
    MAX_LINF_DELTA,
    MIN_LINF_DELTA,
    MIN_PSNR_DB,
    MIN_SSIM,
    RMSE_COMPONENT_WEIGHT,
)
from perturb_mirror.model import (
    LABELS,
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
    """Validator's exact pipeline: resize->480, center-crop->480, BICUBIC,
    normalize with mean=std=0.5 (NOT ImageNet stats). Reuses PREPROCESS."""
    return PREPROCESS(image_bchw)


def compute_preprocess_alive_mask(
    image_bchw: torch.Tensor,
    *,
    grad_eps: float = 1e-9,
) -> torch.Tensor:
    """Pixels that influence EfficientNet through validator PREPROCESS.

    After resize+center-crop, border pixels can be discarded entirely; they
    receive zero gradient and cannot help a flip, but still inflate RMSE if
    perturbed. Returns a B×1×H×W float mask (1.0 = alive, 0.0 = dead).
    """
    x = image_bchw.detach().clone().requires_grad_(True)
    y = apply_validator_preprocess(x)
    y.sum().backward()
    mag = x.grad.abs().amax(dim=1, keepdim=True)
    return (mag > grad_eps).to(dtype=image_bchw.dtype, device=image_bchw.device)


def apply_alive_mask(delta_bchw: torch.Tensor, alive_b1hw: torch.Tensor | None) -> torch.Tensor:
    if alive_b1hw is None:
        return delta_bchw
    return delta_bchw * alive_b1hw


# Validator PREPROCESS uses mean=std=0.5 (see perturb_mirror.model.PREPROCESS / WEIGHTS.transforms())
def _preprocess_mean_std(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    from perturb_mirror.model import PREPROCESS

    mean = torch.tensor(PREPROCESS.mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(PREPROCESS.std, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def denormalize_effnet_preprocess(normalized_bchw: torch.Tensor) -> torch.Tensor:
    """Invert validator PREPROCESS normalization back to RGB (may exceed [0, 1] slightly)."""
    mean, std = _preprocess_mean_std(normalized_bchw.device, normalized_bchw.dtype)
    return normalized_bchw * std + mean


def extract_effnet_feed_bchw(image_bchw: torch.Tensor) -> torch.Tensor:
    """480×480 RGB patch that enters EfficientNet after resize + center crop.

    Uses the same geometry as ``PREPROCESS`` (resize 480, center crop 480, BICUBIC)
    on pixel values in [0, 1], *before* normalization. Do not denormalize the
    normalized tensor for visualization — clamping after denorm washes out colors.
    """
    from torchvision.transforms import functional as TF

    out: list[torch.Tensor] = []
    for i in range(int(image_bchw.shape[0])):
        x = TF.resize(
            image_bchw[i],
            PREPROCESS.resize_size,
            interpolation=PREPROCESS.interpolation,
            antialias=True,
        )
        x = TF.center_crop(x, PREPROCESS.crop_size)
        out.append(x.unsqueeze(0))
    return torch.cat(out, dim=0)


def alive_only_input_bchw(image_bchw: torch.Tensor, alive_b1hw: torch.Tensor | None = None) -> torch.Tensor:
    """Full-resolution input with dead PREPROCESS pixels zeroed (alive footprint only)."""
    if alive_b1hw is None:
        alive_b1hw = compute_preprocess_alive_mask(image_bchw)
    return image_bchw * alive_b1hw


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


def label_index_to_string(idx: int) -> str:
    """Map a class index to the same normalized string predict_label() returns."""
    if 0 <= idx < len(LABELS):
        return normalize_prediction_label(LABELS[idx])
    return str(idx)


def indices_to_label_strings(indices: torch.Tensor) -> list[str]:
    return [label_index_to_string(int(i)) for i in indices.tolist()]


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


# ─── Reward-aligned loss ──────────────────────────────────────────────────────
#
# The validator reward (when every gate passes and SPEED_WEIGHT == 0) is:
#
#     score = 0.7 * (1 - linf_ratio)^2 + 0.3 * (1 - rmse_ratio)^2
#       linf_ratio = (linf - MIN_LINF_DELTA) / (effective_max - MIN_LINF_DELTA)
#       rmse_ratio = rmse / effective_max
#
# Because epsilon is always >= 0.06 > MAX_LINF_DELTA, effective_max == MAX_LINF_DELTA
# in practice, so the ranking is won by the SMALLEST L-inf just above the 0.003
# floor that still flips the label (with SSIM >= 0.98 and PSNR >= 38 dB).
#
# This loss optimises that objective directly instead of the gate proxies.


def _cw_per_sample(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    confidence: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cw_hinge_per_sample, flipped_mask) — flipped == argmax != true."""
    idx = target_indices.view(-1, 1)
    target_logits = logits.gather(1, idx).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, idx, False)
    other_logits = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    cw = torch.clamp(target_logits - other_logits + confidence, min=0.0)
    flipped = (other_logits > target_logits).float()
    return cw, flipped


def ssim_clean_stats(x_clean: torch.Tensor, kernel_size: int = 11) -> dict:
    """Precompute the clean-image SSIM terms that don't depend on the adv image.

    ``_ssim_per_image`` recomputes ``mu_x`` and ``sigma_x`` (2 of 5 pooling ops)
    from the constant clean image on every call. When SSIM is evaluated many
    times against the same clean image (the OPS search), precompute these once
    and feed them to ``_ssim_per_image_cached``.
    """
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(x_clean, kernel_size, stride=1, padding=padding)
    mu_x_sq = mu_x * mu_x
    sigma_x = F.avg_pool2d(x_clean * x_clean, kernel_size, stride=1, padding=padding) - mu_x_sq
    return {
        "x_clean": x_clean,
        "kernel_size": kernel_size,
        "mu_x": mu_x,
        "mu_x_sq": mu_x_sq,
        "sigma_x": sigma_x,
    }


def _ssim_per_image_cached(stats: dict, x_adv: torch.Tensor) -> torch.Tensor:
    """Per-image SSIM (B,) reusing precomputed clean stats from ``ssim_clean_stats``."""
    kernel_size = stats["kernel_size"]
    padding = kernel_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    x_clean = stats["x_clean"]
    mu_x = stats["mu_x"]
    mu_y = F.avg_pool2d(x_adv, kernel_size, stride=1, padding=padding)
    sigma_y = F.avg_pool2d(x_adv * x_adv, kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x_clean * x_adv, kernel_size, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (stats["mu_x_sq"] + mu_y * mu_y + c1) * (stats["sigma_x"] + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return ssim_map.mean(dim=[1, 2, 3])


def _ssim_per_image(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    kernel_size: int = 11,
) -> torch.Tensor:
    """Differentiable per-image SSIM (B,) using the validator formula."""
    return _ssim_per_image_cached(ssim_clean_stats(x_clean, kernel_size), x_adv)


def reward_aligned_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    clean_bchw: torch.Tensor,
    adv_quant: torch.Tensor,
    *,
    min_delta: float = MIN_LINF_DELTA,
    max_delta: float = MAX_LINF_DELTA,
    linf_component_weight: float = 0.7,
    rmse_component_weight: float = 0.3,
    cw_confidence: float = 2.0,
    floor_margin: float = 0.0005,
    linf_topk: int = 32,
    w_flip: float = 1.0,
    w_score: float = 4.0,
    w_floor: float = 80.0,
    w_ssim: float = 60.0,
    w_psnr: float = 0.05,
    min_ssim: float = MIN_SSIM,
    min_psnr_db: float = MIN_PSNR_DB,
    flip_loss_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Loss that directly maximises the validator's perturbation_score.

    Strategy (flip-gated curriculum):
      * `flip` term (margin-CW) is always active so every image learns to flip.
      * The size/quality terms (`score`, `floor`, `ssim`, `psnr`) are applied
        ONLY to images that currently flip (detached mask). Early on, nothing
        flips so the generator focuses on flipping; once an image is adversarial
        the loss squeezes its L-inf toward the 0.003 floor (max reward) while
        holding it above the floor and keeping SSIM/PSNR above their gates.

    L-inf gradient uses a straight-through estimator: forward value is the true
    per-image max, gradient flows through the mean of the top-k abs deltas so
    many pixels are nudged instead of a single one.

    `flip_loss_override` swaps the flip driver: when given, that scalar replaces
    the margin-CW term (and `w_flip` is *not* re-applied — fold any weight into
    it). The LTP feature-separation driver is passed here as a pre-weighted,
    pre-negated distance so the optimiser still pushes adv away from clean
    features. The flipped mask (used only to gate the size/quality terms) still
    comes from the classifier logits.
    """
    cw, flipped = _cw_per_sample(logits, target_indices, cw_confidence)
    cw_mean = cw.mean()
    flip_loss = cw_mean if flip_loss_override is None else flip_loss_override

    delta = adv_quant - clean_bchw
    abs_delta = delta.abs()
    flat = abs_delta.flatten(1)

    linf_hard = flat.amax(dim=1)
    k = min(int(linf_topk), int(flat.shape[1]))
    linf_soft = flat.topk(k, dim=1).values.mean(dim=1)
    # straight-through: value == hard max, gradient == d(top-k mean)
    linf_ste = linf_hard.detach() + (linf_soft - linf_soft.detach())

    rmse = torch.sqrt(delta.pow(2).flatten(1).mean(dim=1) + 1e-12)

    denom = max(1e-12, float(max_delta) - float(min_delta))
    linf_ratio = ((linf_ste - float(min_delta)) / denom).clamp(0.0, 1.0)
    rmse_ratio = (rmse / float(max_delta)).clamp(0.0, 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_score = (1.0 - rmse_ratio) ** 2
    total_w = max(1e-12, float(linf_component_weight) + float(rmse_component_weight))
    pert_score = (
        float(linf_component_weight) * linf_score + float(rmse_component_weight) * rmse_score
    ) / total_w

    floor_hinge = torch.clamp(float(min_delta) + float(floor_margin) - linf_hard, min=0.0)

    ssim = _ssim_per_image(clean_bchw, adv_quant)
    ssim_hinge = torch.clamp(float(min_ssim) - ssim, min=0.0)

    mse = delta.pow(2).flatten(1).mean(dim=1)
    psnr_db = -10.0 * torch.log10(mse + 1e-12)
    psnr_hinge = torch.clamp(float(min_psnr_db) - psnr_db, min=0.0)

    # Masked mean over currently-flipping images (detached gate).
    mask = flipped.detach()
    mask_denom = mask.sum().clamp(min=1.0)

    def masked_mean(term: torch.Tensor) -> torch.Tensor:
        return (term * mask).sum() / mask_denom

    score_loss = masked_mean(1.0 - pert_score)
    floor_loss = masked_mean(floor_hinge)
    ssim_loss = masked_mean(ssim_hinge)
    psnr_loss = masked_mean(psnr_hinge)

    flip_term = flip_loss if flip_loss_override is not None else w_flip * flip_loss
    loss = (
        flip_term
        + w_score * score_loss
        + w_floor * floor_loss
        + w_ssim * ssim_loss
        + w_psnr * psnr_loss
    )

    components = {
        "loss": float(loss.item()),
        "flip": float(flip_loss.item()),
        "score": float(score_loss.item()),
        "floor": float(floor_loss.item()),
        "ssim_h": float(ssim_loss.item()),
        "psnr_h": float(psnr_loss.item()),
        "flip_rate": float(mask.mean().item()),
        "pert_score": float(masked_mean(pert_score).item()),
        "linf_mean": float(linf_hard.mean().item()),
    }
    return loss, components


def mae_aligned_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    clean_bchw: torch.Tensor,
    adv_quant: torch.Tensor,
    *,
    min_delta: float = MIN_LINF_DELTA,
    max_delta: float = MAX_LINF_DELTA,
    linf_component_weight: float = 0.7,
    rmse_component_weight: float = 0.3,
    cw_confidence: float = 6.0,
    floor_margin: float = 0.0005,
    w_flip: float = 1.0,
    w_mae: float = 100.0,
    w_floor: float = 80.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Margin-CW flip driver + MAE perturbation restriction (flip-gated).

        loss = w_flip  * margin_CW
             + w_mae   * MAE(|adv_quant - clean|)   [flipped images only]
             + w_floor * floor_hinge(L-inf < floor) [flipped images only]

    The 1/255 L-inf *upper* bound is enforced architecturally by the generator
    (``tanh(h) * max_linf``), so there is no explicit L-inf ceiling term here.
    The MAE term squeezes the whole perturbation toward zero; the floor hinge
    keeps the per-image L-inf above the validator's MIN_LINF_DELTA so MAE can't
    drive it into the too-small disqualification region.

    Like ``reward_aligned_loss`` this is a flip-gated curriculum: the margin-CW
    term is always active so every image learns to flip, while the size terms
    (MAE, floor) apply only to images that currently flip (detached mask).

    ``pert_score`` (the validator reward) is reconstructed for logging and
    checkpoint selection only; it is not part of the optimised objective.
    """
    cw, flipped = _cw_per_sample(logits, target_indices, cw_confidence)
    cw_mean = cw.mean()

    delta = adv_quant - clean_bchw
    abs_delta = delta.abs()
    flat = abs_delta.flatten(1)

    # Restriction term: per-image mean absolute perturbation.
    mae = flat.mean(dim=1)

    # Hard per-image L-inf for the floor hinge (no STE needed — gradient flows
    # through the MAE term, which already touches every pixel).
    linf_hard = flat.amax(dim=1)
    floor_hinge = torch.clamp(float(min_delta) + float(floor_margin) - linf_hard, min=0.0)

    # Validator reward reconstruction (monitoring / checkpoint proxy only).
    rmse = torch.sqrt(delta.pow(2).flatten(1).mean(dim=1) + 1e-12)
    denom = max(1e-12, float(max_delta) - float(min_delta))
    linf_ratio = ((linf_hard - float(min_delta)) / denom).clamp(0.0, 1.0)
    rmse_ratio = (rmse / float(max_delta)).clamp(0.0, 1.0)
    total_w = max(1e-12, float(linf_component_weight) + float(rmse_component_weight))
    pert_score = (
        float(linf_component_weight) * (1.0 - linf_ratio) ** 2
        + float(rmse_component_weight) * (1.0 - rmse_ratio) ** 2
    ) / total_w

    mask = flipped.detach()
    mask_denom = mask.sum().clamp(min=1.0)

    def masked_mean(term: torch.Tensor) -> torch.Tensor:
        return (term * mask).sum() / mask_denom

    mae_loss = masked_mean(mae)
    floor_loss = masked_mean(floor_hinge)

    loss = w_flip * cw_mean + w_mae * mae_loss + w_floor * floor_loss

    components = {
        "loss": float(loss.item()),
        "flip": float(cw_mean.item()),
        "mae": float(mae_loss.item()),
        "floor": float(floor_loss.item()),
        "flip_rate": float(mask.mean().item()),
        "pert_score": float(masked_mean(pert_score).item()),
        "linf_mean": float(linf_hard.mean().item()),
    }
    return loss, components


# ─── LTP mid-level feature separation ─────────────────────────────────────────
#
# Learning Transferable Adversarial Perturbations (Nakka & Salzmann, NeurIPS'21)
# drives the attack by *maximising* the distance between the clean and adversarial
# mid-level feature maps of a frozen classifier (a label-free, smooth flip
# signal). EfficientNetV2-L exposes its blocks as `model.features` (a Sequential
# of 9 stages, indices 0..8); we tap one or more of them (mid-level ~3-5).


def efficientnet_num_feature_blocks(model: torch.nn.Module) -> int:
    """Number of `model.features` stages (9 for EfficientNetV2-L)."""
    core = getattr(model, "_orig_mod", model)
    return len(core.features) if hasattr(core, "features") else 0


def _features_and_logits(
    model: torch.nn.Module,
    model_in: torch.Tensor,
    feat_layers: tuple[int, ...],
    grad_checkpoint: bool = False,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Single pass that returns logits AND the requested intermediate maps.

    Walks `model.features` block by block, collecting outputs at `feat_layers`,
    then finishes avgpool -> flatten -> classifier for the logits.
    """
    from torch.utils.checkpoint import checkpoint

    core = getattr(model, "_orig_mod", model)
    if not hasattr(core, "features"):
        return model(model_in), {}

    wanted = set(int(i) for i in feat_layers)
    feats: dict[int, torch.Tensor] = {}
    use_ckpt = grad_checkpoint and torch.is_grad_enabled()

    out = model_in
    for i, block in enumerate(core.features):
        out = checkpoint(block, out, use_reentrant=False) if use_ckpt else block(out)
        if i in wanted:
            feats[i] = out

    pooled = torch.flatten(core.avgpool(out), 1)
    logits = core.classifier(pooled)
    return logits, feats


@torch.no_grad()
def clean_feature_maps(
    model: torch.nn.Module,
    clean_bchw: torch.Tensor,
    feat_layers: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Detached clean mid-level features (validator preprocess), stop-early."""
    core = getattr(model, "_orig_mod", model)
    x = apply_validator_preprocess(clean_bchw)
    if not hasattr(core, "features"):
        return {}
    wanted = set(int(i) for i in feat_layers)
    last = max(wanted) if wanted else -1
    feats: dict[int, torch.Tensor] = {}
    for i, block in enumerate(core.features):
        x = block(x)
        if i in wanted:
            feats[i] = x.detach()
        if i >= last:
            break
    return feats


def feat_separation_loss(
    feats_adv: dict[int, torch.Tensor],
    feats_clean: dict[int, torch.Tensor],
    normalize: bool = True,
) -> torch.Tensor:
    """Summed feature distance to MAXIMISE (LTP `feat_loss_mutliscale_fn`).

    Per layer: MSE(adv, clean). With `normalize`, divide by the clean feature
    energy so each tapped layer contributes on a comparable ~O(1) scale,
    regardless of its activation magnitude (makes `w_feat` tuning stable).
    Returns a positive distance; the trainer negates it for the flip driver.
    """
    total: torch.Tensor | None = None
    for k in feats_adv:
        a = feats_adv[k]
        c = feats_clean[k]
        d = (a - c).pow(2).mean()
        if normalize:
            d = d / (c.pow(2).mean() + 1e-8)
        total = d if total is None else total + d
    if total is None:
        # No features tapped — return a graph-connected zero.
        any_feat = next(iter(feats_adv.values()), None)
        if any_feat is not None:
            return any_feat.sum() * 0.0
        return torch.zeros((), device=feats_clean[next(iter(feats_clean))].device) \
            if feats_clean else torch.zeros(())
    return total


# ─── Full forward pass ────────────────────────────────────────────────────────

def _classifier_forward_checkpointed(
    model: torch.nn.Module,
    x: torch.Tensor,
    segments: int = 4,
) -> torch.Tensor:
    """EfficientNet forward with gradient checkpointing over the feature blocks.

    Activations of `model.features` are dropped on the forward pass and
    recomputed during backward, trading compute for a large VRAM reduction.
    Safe because the classifier is frozen + in eval mode (BN uses running stats,
    dropout is identity), so the recomputation is deterministic.
    """
    from torch.utils.checkpoint import checkpoint_sequential

    core = getattr(model, "_orig_mod", model)  # unwrap torch.compile if present
    if not hasattr(core, "features"):
        return model(x)  # unknown architecture -> no checkpointing

    x = checkpoint_sequential(core.features, segments, x, use_reentrant=False)
    x = core.avgpool(x)
    x = torch.flatten(x, 1)
    return core.classifier(x)


def forward_adv(
    model: torch.nn.Module,
    generator: torch.nn.Module,
    clean_bchw: torch.Tensor,
    channels_last: bool = False,
    grad_checkpoint: bool = False,
    checkpoint_segments: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generator → STE quantize → PREPROCESS → logits.
    Pipeline:  clean [0,1] → δ → adv → uint8 STE → PREPROCESS → EfficientNet
    Returns (logits, adv_quant). adv_quant stays in NCHW [0,1] for the loss.
    """
    perturbation = generator(clean_bchw)
    adv = torch.clamp(clean_bchw + perturbation, 0.0, 1.0)
    adv_quant = quantize_ste(adv)
    model_in = apply_validator_preprocess(adv_quant)
    if channels_last:
        model_in = model_in.contiguous(memory_format=torch.channels_last)
    if grad_checkpoint and torch.is_grad_enabled():
        logits = _classifier_forward_checkpointed(model, model_in, checkpoint_segments)
    else:
        logits = model(model_in)
    return logits, adv_quant


def forward_adv_with_feats(
    model: torch.nn.Module,
    generator: torch.nn.Module,
    clean_bchw: torch.Tensor,
    feat_layers: tuple[int, ...],
    channels_last: bool = False,
    grad_checkpoint: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """Like :func:`forward_adv` but also returns adversarial mid-level features.

    Used by the LTP feature-separation flip driver. Logits and features come
    from a single classifier pass over the (STE-quantised, preprocessed) adv
    image, so the extra signal costs no additional forward.
    """
    perturbation = generator(clean_bchw)
    adv = torch.clamp(clean_bchw + perturbation, 0.0, 1.0)
    adv_quant = quantize_ste(adv)
    model_in = apply_validator_preprocess(adv_quant)
    if channels_last:
        model_in = model_in.contiguous(memory_format=torch.channels_last)
    logits, feats = _features_and_logits(model, model_in, feat_layers, grad_checkpoint)
    return logits, adv_quant, feats


# ─── Validator-grade evaluation (no grad) ─────────────────────────────────────

@torch.no_grad()
def eval_batch(
    model: torch.nn.Module,
    generator: torch.nn.Module,
    clean_bchw: torch.Tensor,
    true_indices: torch.Tensor,
    epsilon: float,
    min_linf_delta: float = MIN_LINF_DELTA,
    min_ssim: float = MIN_SSIM,
    min_psnr_db: float = MIN_PSNR_DB,
) -> dict[str, float]:
    """
    Vectorized validator-gate check over a same-shape batch (fp32, validator-exact).
    Gates: min_delta, max_delta, label_flip, min_ssim, min_psnr_db. The adversarial
    prediction is a single batched forward (vs one forward per image previously).
    """
    batch = int(clean_bchw.shape[0])
    if batch == 0:
        return {"pass_rate": 0.0, "flip_rate": 0.0, "ssim_mean": 0.0, "psnr_mean": 0.0,
                "score_mean": 0.0}

    perturbation = generator(clean_bchw)
    adv = torch.clamp(clean_bchw + perturbation, 0.0, 1.0)
    adv_quant = quantize_ste(adv)

    delta = adv_quant - clean_bchw
    flat = delta.abs().flatten(1)
    linf = flat.amax(dim=1)
    mse = delta.pow(2).flatten(1).mean(dim=1)
    rmse = torch.sqrt(mse)
    psnr = torch.where(mse <= 1e-12, torch.full_like(mse, 99.0), 10.0 * torch.log10(1.0 / (mse + 1e-12)))
    ssim = _ssim_per_image(clean_bchw, adv_quant)

    pred_idx = model(apply_validator_preprocess(adv_quant)).argmax(dim=1)
    flipped = pred_idx != true_indices.to(pred_idx.device)

    effective_max_delta = min(epsilon, MAX_LINF_DELTA)
    gate = (
        (linf >= min_linf_delta)
        & (linf <= effective_max_delta)
        & (ssim >= min_ssim)
        & ((psnr >= min_psnr_db) if min_psnr_db > 0.0 else torch.ones_like(flipped))
        & flipped
    )

    denom = max(1e-12, effective_max_delta - min_linf_delta)
    linf_ratio = ((linf - min_linf_delta) / denom).clamp(0.0, 1.0)
    rmse_ratio = (rmse / effective_max_delta).clamp(0.0, 1.0)
    total_w = max(1e-12, LINF_COMPONENT_WEIGHT + RMSE_COMPONENT_WEIGHT)
    pert_score = (
        LINF_COMPONENT_WEIGHT * (1.0 - linf_ratio) ** 2
        + RMSE_COMPONENT_WEIGHT * (1.0 - rmse_ratio) ** 2
    ) / total_w
    score = torch.where(gate, pert_score, torch.zeros_like(pert_score))

    return {
        "pass_rate": float(gate.float().mean().item()),
        "flip_rate": float(flipped.float().mean().item()),
        "ssim_mean": float(ssim.mean().item()),
        "psnr_mean": float(psnr.mean().item()),
        "score_mean": float(score.mean().item()),
    }
