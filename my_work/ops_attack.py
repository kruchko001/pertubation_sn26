"""Wrapper around the OPS attack (OPS/TransferAttack) pinned to EfficientNetV2-L.

OPS (Operator-Perturbation-based Stochastic optimization) is designed for
*transferable* attacks: it averages gradients over random image operators and
neighbor perturbations so a perturbation crafted on a surrogate transfers to
unknown black-box models.

For subnet26 the target model is known and fixed (EfficientNetV2-L), so there is
nothing to transfer to. This wrapper points OPS directly at EfficientNetV2-L
behind the validator's *exact* preprocess pipeline, turning OPS into a white-box
attack that overfits to EfficientNetV2-L instead of optimising for transfer.

Two knobs control the transfer-vs-overfit trade-off (see build_ops_attack):
  * num_sample_operator / num_sample_neighbor > 0  -> full OPS (transfer-robust)
  * either set to 0                                -> sampling off == MI-FGSM,
                                                      pure white-box overfit.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from model_utils import (
    _ssim_per_image_cached,
    apply_alive_mask,
    apply_validator_preprocess,
    compute_preprocess_alive_mask,
    quantize_ste,
    ssim_clean_stats,
)
from transform_ops import (
    N_L2T_OPS,
    apply_aitl_chain,
    apply_l2t_op,
    cw_margin_loss,
    dlr_margin_loss,
    l2t_select_ops,
    l2t_trace_prob,
    sample_aitl_chain,
)
from perturb_mirror.constants import (
    MAX_LINF_DELTA,
    MIN_LINF_DELTA,
    MIN_PSNR_DB,
    MIN_SSIM,
)
from perturb_mirror.model import load_efficientnet_v2_l

# OPS repo lives at <project_root>/OPS/TransferAttack (sibling of my_work).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OPS_PKG_ROOT = _PROJECT_ROOT / "OPS" / "TransferAttack"

# Largest L-inf budget that still satisfies the validator's max-delta gate AFTER
# uint8 PNG rounding: a float delta d encodes to round(d * 255) / 255, so the
# biggest safe integer step is floor(MAX_LINF_DELTA * 255) / 255 (= 7/255 for
# the default 0.03 cap). Using exactly MAX_LINF_DELTA rounds up to 8/255 > 0.03.
QUANT_SAFE_MAX_LINF = math.floor(MAX_LINF_DELTA * 255.0) / 255.0


def _import_ops_class():
    """Import the upstream OPS class without modifying the OPS repo."""
    if not _OPS_PKG_ROOT.is_dir():
        raise FileNotFoundError(
            f"OPS repo not found at {_OPS_PKG_ROOT}. Expected OPS/TransferAttack "
            "to exist at the project root."
        )
    if str(_OPS_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_OPS_PKG_ROOT))
    from transferattack.input_transformation.ops import OPS  # noqa: E402

    return OPS


class EfficientNetV2LSurrogate(nn.Module):
    """Validator-exact classifier head for OPS.

    OPS calls ``self.model(x)`` on images in pixel space ``[0, 1]``. The Perturb
    validator feeds the perturbed image through ``WEIGHTS.transforms()`` (resize
    to 480 + center crop + normalize mean=std=0.5) before EfficientNetV2-L, so we
    fold that exact preprocess in here. This makes OPS gradients overfit to the
    same model+pipeline that scores the submission.

    Note: normalization is part of PREPROCESS, so unlike OPS's default
    ``wrap_model`` we do NOT add a separate Normalize layer (no double-norm).
    """

    def __init__(self, device: torch.device, *, channels_last: bool = True) -> None:
        super().__init__()
        self.device = device
        self.model = load_efficientnet_v2_l(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        # The OPS pipeline runs a fixed 480x480 input through this head thousands
        # of times, so enable the same conv-throughput flags the trainer uses.
        self._channels_last = bool(channels_last) and device.type == "cuda"
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        if self._channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = apply_validator_preprocess(x)
        if self._channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.model(x)


def build_ops_attack(
    device: torch.device,
    *,
    epsilon: float = QUANT_SAFE_MAX_LINF,
    num_iter: int = 10,
    num_sample_neighbor: int = 10,
    num_sample_operator: int = 20,
    beta: float = 2.0,
    decay: float = 1.0,
    targeted: bool = False,
    random_start: bool = False,
    surrogate: EfficientNetV2LSurrogate | None = None,
):
    """Construct an OPS attack instance bound to EfficientNetV2-L.

    Args:
        device: torch device for the model and tensors.
        epsilon: L-inf budget. Defaults to QUANT_SAFE_MAX_LINF (7/255) so that
            after uint8 PNG rounding the perturbation still satisfies the
            validator's max-delta gate (effective cap min(challenge_eps, 0.03)).
        num_iter: OPS iterations (alpha = epsilon / num_iter).
        num_sample_neighbor: neighbor perturbation samples per step.
        num_sample_operator: random-operator samples per neighbor.
            Set either *_sample_* to 0 for pure white-box MI-FGSM (overfit).
        beta, decay: OPS neighbor radius scale and momentum decay.
        targeted: targeted attack (expects [gt, target] labels) if True.
        random_start: random delta init.
        surrogate: reuse an existing surrogate (avoids reloading weights).

    Returns:
        An OPS instance whose ``__call__(data, label)`` returns delta.
        The bound classifier head is available as ``attack.model``.
    """
    OPS = _import_ops_class()
    head = surrogate if surrogate is not None else EfficientNetV2LSurrogate(device)

    class OPSEfficientNetV2L(OPS):
        """OPS pinned to the validator-exact EfficientNetV2-L head."""

        def load_model(self, model_name):  # noqa: ARG002 - name ignored on purpose
            return head

    attack = OPSEfficientNetV2L(
        model_name="efficientnet_v2_l",  # ignored by overridden load_model
        epsilon=float(epsilon),
        beta=float(beta),
        num_iter=int(num_iter),
        num_sample_neighbor=int(num_sample_neighbor),
        num_sample_operator=int(num_sample_operator),
        decay=float(decay),
        targeted=bool(targeted),
        random_start=bool(random_start),
        norm="linfty",
        loss="crossentropy",
        device=device,
    )
    return attack


def forwards_per_image(
    num_iter: int,
    num_sample_neighbor: int,
    num_sample_operator: int,
) -> int:
    """Estimate classifier forward passes per image (for cost reporting)."""
    if num_sample_neighbor * num_sample_operator > 0:
        per_step = num_sample_neighbor * num_sample_operator + 1
    else:
        per_step = 1
    return int(num_iter) * per_step


# ─── Budget-minimizing (L-inf / SSIM-aware) objective ──────────────────────────
#
# Vanilla OPS only maximises cross-entropy: it flips the label but spends the
# whole L-inf budget, which tanks SSIM/PSNR and wins zero validator reward.
#
# The validator reward (gates passed, SPEED_WEIGHT == 0) is dominated by the
# SMALLEST L-inf just above the 0.003 floor that still flips, with SSIM >= 0.98
# and PSNR >= 38 dB. So we want a *minimum-norm* adversarial example, not a
# max-confidence one.
#
# OPSBudgetMin keeps OPS's (optionally operator/neighbor-sampled, model-overfit)
# gradient as the flip direction, and adds a differentiable objective that:
#   * shrinks L-inf (top-k STE) and RMSE once the sample already flips,
#   * holds L-inf above the min-delta floor (avoids `below_min_delta`),
#   * lifts SSIM and PSNR over their gates (hinge penalties),
# all evaluated on the uint8-quantized image so the in-loop gates match the
# validator byte-for-byte. The smallest passing iterate is returned per image.


@dataclass
class BudgetConfig:
    """Highest-score (minimum-norm) objective for the validator reward.

    The reward (gates passed, SPEED_WEIGHT == 0) is
        0.7 * (1 - (linf - 0.003)/(eff_max - 0.003))**2  +  0.3 * (1 - rmse/eff_max)**2
    with eff_max == 0.03. After uint8 rounding L-inf is an integer number of
    1/255 steps, so there are only 7 feasible L-inf levels (1/255 .. 7/255). The
    score is maximised by the SMALLEST level that flips (L-inf term) and the
    FEWEST changed pixels at that level (RMSE term). Strategy:

      1. Ascending-scan the smallest feasible level in {1..max_level}/255,
         running a strong PGD (many iters + momentum + restarts) at each level.
         Level 1 gets `first_level_restarts` extra random restarts because it is
         0.7-weighted and worth ~0.2 over level 2, so even hard images flip at
         1/255 rather than slipping to 2/255.
      2. Sparse L0 refinement at the chosen level: binary-search the minimum
         support (re-optimised per k, with restarts) that keeps the flip, then a
         greedy backward-elimination polish drops any remaining redundant pixels.
         Both cut RMSE, which (at fixed 1/255) is sqrt(k/N)/255.

    A logit ``flip_margin`` buffer makes solutions survive the validator's own
    (independent, slightly non-deterministic) forward pass.
    """

    max_level: int = 1          # max L-inf in 1/255 units (floor(0.03*255) == 7)
    inner_iters: int = 15       # PGD steps per level
    restarts: int = 2           # PGD restarts per level (>=1; >1 adds random init)
    first_level_restarts: int = 2  # extra restarts at level 1 (the 0.7-weighted floor)
    alpha_frac: float = 0.5     # PGD step size as a fraction of eps (>= 1/255)
    decay: float = 1.0          # momentum decay
    min_delta: float = MIN_LINF_DELTA
    min_ssim: float = MIN_SSIM
    min_psnr_db: float = MIN_PSNR_DB
    eff_max_eps: float = MAX_LINF_DELTA
    flip_margin: float = 2.0    # logit margin required when PRUNING (conservative)
    accept_margin: float = 0.5  # logit margin to accept an L-inf level (see note)
    sparse: bool = True
    sparse_steps: int = 15      # PGD steps per support size in the k binary search
    sparse_restarts: int = 2    # restarts of the support-restricted PGD (>1 random)
    polish_rounds: int = 3      # backward-elimination passes after the k search
    crop_mask: bool = True      # zero delta outside PREPROCESS alive footprint
    # ── Loss function for the PGD ascent direction ───────────────────────────
    loss_type: str = "cw"             # "cw" = C&W logit margin | "dlr" = Difference-of-Logits-Ratio
    loss_target_margin: float | None = None  # cap each row's CW margin at kappa (None = no cap; ignored for dlr)
    # ── Gradient diversity (AITL / L2T) ──────────────────────────────────────
    grad_diversity: int = 0           # 0 = plain CW grad; N = average over N augmented inputs
    grad_transform_src: str = "aitl"  # "aitl" | "l2t"
    aitl_chain_len: int = 4           # ops per AITL random chain
    l2t_n_ops: int = 2                # ops per L2T chain
    l2t_lr: float = 0.01             # L2T aug_param learning rate
    # ── White-box adaptations ─────────────────────────────────────────────────
    grad_smooth_dct: bool = False     # low-pass gradient in DCT domain before momentum
    grad_smooth_frac: float = 0.5     # fraction of DCT frequencies to keep (0 = DC only, 1 = all)
    aitl_restarts: bool = False       # seed PGD restarts r>0 with AITL-guided gradient direction
    # ── DualMIFGSM / Ens-FGSM-MIFGSM tricks ─────────────────────────────────
    dual_example: bool = False        # compute each step's gradient at a fresh random δ, not current δ
    dual_ensemble: int = 1            # average gradients from this many random δ per step (1=Dual, N=Ensemble)
    # ── Sign-of-Adam (AMI-FGSM style) ────────────────────────────────────────
    use_adam: bool = False            # replace MI-FGSM momentum with sign(Adam direction)
    adam_beta1: float = 0.9           # Adam first-moment decay
    adam_beta2: float = 0.999         # Adam second-moment decay
    adam_eps: float = 1e-8            # Adam numerical stability


def budget_config_fast53() -> BudgetConfig:
    """~53 s/image preset (ops_bench_50_cw.log: mean 52.8 s, score 0.933)."""
    return BudgetConfig(
        sparse_steps=30,
        sparse_restarts=1,
        polish_rounds=3,
        first_level_restarts=2,
    )


def budget_config_adam() -> BudgetConfig:
    """Fast53 + sign-of-Adam gradient direction (AMI-FGSM style).

    Replaces the MI-FGSM momentum accumulation with Adam's variance-normalised
    direction, then applies the same sign step.  Per-pixel variance weighting
    suppresses noisy gradient dimensions and amplifies consistent ones, which
    can improve pixel selection without altering the L-inf constraint or step
    size.  beta1=0.9, beta2=0.999, bias correction applied each step.
    """
    cfg = budget_config_fast53()
    cfg.use_adam = True
    return cfg


def budget_config_cw_margin(kappa: float = 2.0) -> BudgetConfig:
    """Fast53 + C&W confidence kappa baked into the ascent loss.

    Each row's CW margin is capped at ``kappa`` so PGD stops pushing a sample
    once it clears the buffer and spends its remaining budget on rows that have
    not yet flipped past ``kappa``.  Pairing ``kappa`` with cfg.flip_margin (the
    pruning buffer) means more restarts pass on the first try at a given level,
    cutting restart spend.  Default kappa = flip_margin = 2.0.
    """
    cfg = budget_config_fast53()
    cfg.loss_type = "cw"
    cfg.loss_target_margin = kappa
    return cfg


def budget_config_dlr() -> BudgetConfig:
    """Fast53 + Difference-of-Logits-Ratio ascent loss.

    Replaces the raw C&W margin with DLR: the margin normalised by the spread
    between the 1st and 3rd largest logits.  Scale-invariance mainly helps the
    gradient-averaging / REINFORCE paths (AITL/L2T) where unnormalised margins
    let high-magnitude transform copies dominate; for plain sign-PGD the sign is
    already scale-free, so expect little change there.
    """
    cfg = budget_config_fast53()
    cfg.loss_type = "dlr"
    return cfg


def budget_config_dlr_aitl(n: int = 5) -> BudgetConfig:
    """DLR ascent loss + AITL gradient diversity (N transform chains/step).

    The combination most likely to pay off: DLR's per-row normalisation removes
    the scale bias in the AITL gradient average, so each transform chain
    contributes comparably to the isotropic gradient estimate.
    """
    cfg = budget_config_fast53()
    cfg.loss_type = "dlr"
    cfg.grad_diversity = n
    cfg.grad_transform_src = "aitl"
    return cfg


def budget_config_aitl(n: int = 5) -> BudgetConfig:
    """Fast53 + AITL gradient diversity (N random transform chains per step).

    Each PGD step averages CW gradients from ``n`` independently-sampled
    4-op AITL chains, giving a more isotropic gradient estimate at the cost
    of ~n× more forward passes per step.  Recommended ``n`` = 3..5.
    """
    cfg = budget_config_fast53()
    cfg.grad_diversity = n
    cfg.grad_transform_src = "aitl"
    return cfg


def budget_config_l2t(n: int = 3) -> BudgetConfig:
    """Fast53 + L2T gradient diversity (learned op distribution per image).

    Each PGD step samples ``n`` chains from a per-image softmax over the
    ~98-op L2T op_list, averages CW gradients, and updates the distribution
    via a REINFORCE policy gradient so harder-to-transform directions are
    up-weighted over time.  Recommended ``n`` = 3.
    """
    cfg = budget_config_fast53()
    cfg.grad_diversity = n
    cfg.grad_transform_src = "l2t"
    return cfg


def budget_config_smooth() -> BudgetConfig:
    """Fast53 + DCT gradient low-pass filter (white-box RMSE optimisation).

    Before the MI-FGSM momentum accumulation each step the raw CW gradient is
    projected onto its low spatial-frequency components via rfft2/irfft2 (keeping
    the bottom-half of DCT bins).  This biases delta toward smooth, spatially-
    correlated patterns that touch fewer pixels at the same L-inf → lower RMSE.
    """
    cfg = budget_config_fast53()
    cfg.grad_smooth_dct = True
    cfg.grad_smooth_frac = 0.5
    return cfg


def budget_config_aitl_restarts() -> BudgetConfig:
    """Fast53 + AITL-guided diverse PGD restart initialisation.

    For every restart r>0 the initial delta is set to ``eps * sign(∇CW(T(x)))``
    where T is a freshly-sampled 4-op AITL transform chain.  This costs one
    extra forward pass per non-zero restart (minor) but explores geometrically
    distinct regions of delta-space, improving the probability of finding level-1
    flips on borderline-hard images.
    """
    cfg = budget_config_fast53()
    cfg.aitl_restarts = True
    return cfg


def budget_config_smooth_aitl() -> BudgetConfig:
    """Combination: DCT gradient smoothing + AITL diverse restarts."""
    cfg = budget_config_fast53()
    cfg.grad_smooth_dct = True
    cfg.grad_smooth_frac = 0.5
    cfg.aitl_restarts = True
    return cfg


def budget_config_dual() -> BudgetConfig:
    """Fast53 + DualMIFGSM trick (gradient at random δ each step).

    At each inner PGD step the gradient is evaluated at a freshly-sampled
    random δ ∈ Uniform(-eps, eps) instead of the current δ.  The momentum
    then accumulates this "random-point" gradient, and the main δ is updated
    with its sign.  This separates exploration (random δ) from exploitation
    (sign(momentum)) and averages out curvature artefacts near the boundary.
    """
    cfg = budget_config_fast53()
    cfg.dual_example = True
    cfg.dual_ensemble = 1
    return cfg


def budget_config_dual_ensemble(n: int = 5) -> BudgetConfig:
    """Fast53 + Ens-FGSM-MIFGSM trick (N random-δ gradients averaged per step).

    Extends DualMIFGSM by averaging ``n`` independent random-δ gradients per
    step before momentum accumulation.  Reduces gradient variance at the cost
    of ``n`` forward passes per step.  Recommended n = 3..5.
    """
    cfg = budget_config_fast53()
    cfg.dual_example = True
    cfg.dual_ensemble = n
    return cfg


def _make_budget_forward(cfg: BudgetConfig):
    """Build the highest-score (level-search + sparse) forward() for the config."""

    unit = 1.0 / 255.0
    # Per-image memo, refreshed by _set_image() at the top of each image: the
    # constant true index and the clean-image SSIM terms. Avoids a label .item()
    # sync on every grad call and recomputing mu_x/sigma_x on every _eval.
    _img: dict = {}

    def _set_image(data1, label1):
        _img["idx"] = int(label1.item())
        _img["ssim"] = ssim_clean_stats(data1)

    def _margin(model, data1, delta1):
        """(best_other - true) logit gap on the uint8-quantized image (>0 == flip).

        Skips the SSIM/PSNR/L-inf gate work — for hot paths (polish trials) where
        those gates provably cannot change, so only the margin needs re-checking.
        """
        adv_q = quantize_ste(torch.clamp(data1 + delta1, 0.0, 1.0))
        idx = _img["idx"]
        with torch.inference_mode():
            logits = model(adv_q)[0]
            other = logits.clone()
            other[idx] = float("-inf")
            return float((other.max() - logits[idx]).item())

    def _eval(model, data1, label1, delta1):
        """Validator-exact metrics for one sample on the uint8-quantized image.

        Returns (gates_ok, linf, margin) where gates_ok covers delta/SSIM/PSNR
        bounds and ``margin`` is the (best_other - true) logit gap (>0 == flip).
        """
        adv_q = quantize_ste(torch.clamp(data1 + delta1, 0.0, 1.0))
        dq = adv_q - data1
        linf = float(dq.abs().max().item())
        mse = float(dq.pow(2).mean().item())
        psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        idx = _img["idx"]
        with torch.inference_mode():
            ssim = float(_ssim_per_image_cached(_img["ssim"], adv_q)[0].item())
            logits = model(adv_q)[0]
            other = logits.clone()
            other[idx] = float("-inf")
            margin = float((other.max() - logits[idx]).item())
        gates = (
            (linf >= cfg.min_delta)
            and (linf <= cfg.eff_max_eps)
            and (ssim >= cfg.min_ssim)
            and (psnr >= cfg.min_psnr_db)
        )
        return gates, linf, margin

    def _flipped(model, data1, label1, delta1):
        """True iff the uint8-quantized adversarial argmax differs from label."""
        adv_q = quantize_ste(torch.clamp(data1 + delta1, 0.0, 1.0))
        with torch.inference_mode():
            return int(model(adv_q)[0].argmax().item()) != _img["idx"]

    def budget_forward(self, data: torch.Tensor, label: torch.Tensor, **kwargs):
        device = self.device
        data = data.clone().detach().to(device)
        label = label.clone().detach().to(device)
        model = self.model

        def _mask(delta):
            return apply_alive_mask(delta, alive)

        # ── L2T per-image state (aug_param updated in-place each grad call) ──
        _l2t_aug_param: torch.Tensor | None = None
        if cfg.grad_diversity > 0 and cfg.grad_transform_src == "l2t":
            _l2t_aug_param = torch.zeros(N_L2T_OPS, device=device)

        def _margin_loss(logits, idx: int):
            """Dispatch to the configured ascent objective (maximised to flip).

            Accepts (N, C) or (C,) logits. ``cw`` honours cfg.loss_target_margin
            (a C&W confidence kappa); ``dlr`` is self-normalising so the cap is
            not applicable.
            """
            if cfg.loss_type == "dlr":
                return dlr_margin_loss(logits, idx)
            return cw_margin_loss(logits, idx, target_margin=cfg.loss_target_margin)

        def _grad_plain(data1: torch.Tensor, label1: torch.Tensor,
                        delta: torch.Tensor) -> tuple[torch.Tensor, float]:
            """Logit-margin gradient (white-box, no augmentation).

            Also returns the raw (best_other - true) continuous logit gap at the
            current delta, read off the same forward pass. A positive gap is a
            necessary precondition for a quantized flip, so callers reuse it as a
            cheap gate for the early-stop confirm (no extra forward pass).
            """
            d = delta.detach().requires_grad_(True)
            logits = model(torch.clamp(data1 + d, 0.0, 1.0))
            idx = _img["idx"]
            loss = _margin_loss(logits, idx)
            g = torch.autograd.grad(loss, d)[0].detach()
            row = logits[0].detach()
            other = row.clone()
            other[idx] = float("-inf")
            cont_margin = float((other.max() - row[idx]).item())
            return g, cont_margin

        def _grad_aitl(data1: torch.Tensor, label1: torch.Tensor,
                       delta: torch.Tensor) -> torch.Tensor:
            """Average CW gradient over N random AITL transform chains."""
            idx = _img["idx"]
            grads: list[torch.Tensor] = []
            for _ in range(cfg.grad_diversity):
                chain = sample_aitl_chain(cfg.aitl_chain_len)
                d = delta.detach().requires_grad_(True)
                adv = apply_aitl_chain(torch.clamp(data1 + d, 0.0, 1.0), chain)
                loss = _margin_loss(model(adv), idx)
                grads.append(torch.autograd.grad(loss, d)[0].detach())
            return torch.stack(grads).mean(0)

        def _grad_l2t(data1: torch.Tensor, label1: torch.Tensor,
                      delta: torch.Tensor) -> torch.Tensor:
            """Average CW gradient over N L2T-sampled op chains; update aug_param."""
            nonlocal _l2t_aug_param
            ap = _l2t_aug_param  # type: ignore[assignment]
            idx = _img["idx"]
            grads: list[torch.Tensor] = []
            aug_terms: list[torch.Tensor] = []

            for _ in range(cfg.grad_diversity):
                op_ids = l2t_select_ops(ap, cfg.l2t_n_ops)

                # --- delta gradient (pass 1) ---
                d = delta.detach().requires_grad_(True)
                adv = torch.clamp(data1 + d, 0.0, 1.0)
                for oid in op_ids:
                    adv = apply_l2t_op(adv, oid, max_batch=4)
                loss_val = _margin_loss(model(adv), idx)
                grads.append(torch.autograd.grad(loss_val, d)[0].detach())

                # --- aug_param gradient via REINFORCE (pass 2, stop-grad on loss) ---
                ap_leaf = ap.detach().requires_grad_(True)
                prob = l2t_trace_prob(ap_leaf, op_ids)
                aug_terms.append(prob * loss_val.detach())

            # REINFORCE update: aug_param += lr * ∂(mean(prob*loss))/∂aug_param
            aug_loss = torch.stack(aug_terms).mean()
            aug_grad = torch.autograd.grad(aug_loss, ap_leaf)[0].detach()  # type: ignore[possibly-undefined]
            ap.add_(cfg.l2t_lr * aug_grad)

            return torch.stack(grads).mean(0)

        # _grad returns the gradient only (used everywhere); _grad_m also returns
        # the continuous logit gap when it is taken at the passed delta (plain
        # path) so the inner loop can early-stop without an extra forward. The
        # augmented paths sample grads off transformed inputs, so no usable gap.
        if cfg.grad_diversity == 0:
            def _grad(data1, label1, delta):  # type: ignore[assignment]
                return _grad_plain(data1, label1, delta)[0]

            def _grad_m(data1, label1, delta):  # type: ignore[assignment]
                return _grad_plain(data1, label1, delta)
        elif cfg.grad_transform_src == "l2t":
            def _grad(data1, label1, delta):  # type: ignore[assignment]
                return _grad_l2t(data1, label1, delta)

            def _grad_m(data1, label1, delta):  # type: ignore[assignment]
                return _grad_l2t(data1, label1, delta), None
        else:
            def _grad(data1, label1, delta):  # type: ignore[assignment]
                return _grad_aitl(data1, label1, delta)

            def _grad_m(data1, label1, delta):  # type: ignore[assignment]
                return _grad_aitl(data1, label1, delta), None

        # ── DCT gradient low-pass filter ─────────────────────────────────────
        def _smooth(g: torch.Tensor) -> torch.Tensor:
            """Zero the top (1-grad_smooth_frac) of DCT spatial frequencies.

            Projects the raw CW gradient onto low-frequency components so the
            resulting update perturbs large, spatially-correlated patches rather
            than scattered high-frequency pixels.  This biases the perturbation
            toward fewer unique changed pixels (lower L0/RMSE) at the same L-inf.
            """
            if not cfg.grad_smooth_dct:
                return g
            G = torch.fft.rfft2(g)
            H = G.shape[-2]
            W = G.shape[-1]
            kh = max(1, int(H * cfg.grad_smooth_frac))
            kw = max(1, int(W * cfg.grad_smooth_frac))
            mask = torch.zeros_like(G)
            mask[..., :kh, :kw] = 1.0
            return torch.fft.irfft2(G * mask, s=g.shape[-2:])

        # ── AITL-guided restart initialisation helper ─────────────────────────
        def _aitl_init(data1: torch.Tensor, label1: torch.Tensor,
                       eps: float) -> torch.Tensor:
            """One gradient step under a random AITL transform → diverse init delta.

            Costs a single forward+backward pass.  Returns an eps-clipped delta in
            the sign direction of ∇CW(T(x)), i.e. the steepest ascent direction
            for a randomly-transformed copy of the image.
            """
            chain = sample_aitl_chain(cfg.aitl_chain_len)
            idx = _img["idx"]
            d0 = torch.zeros_like(data1, requires_grad=True)
            adv_t = apply_aitl_chain(torch.clamp(data1 + d0, 0.0, 1.0), chain)
            loss = _margin_loss(model(adv_t), idx)
            g_init = torch.autograd.grad(loss, d0)[0].detach()
            delta = _mask((eps * g_init.sign()).clamp(-eps, eps))
            return _mask(torch.min(torch.max(delta, -data1), 1.0 - data1))

        def _pgd_at_level(data1, label1, level, init=None, restarts=None, target=None):
            """Strong margin-PGD constrained to L-inf = level/255.

            Returns the best delta over restarts, preferring gate-passing
            solutions and then the largest flip margin. ``restarts`` overrides
            cfg.restarts so the caller can spend extra random restarts on the
            most valuable level (1/255), where missing the flip costs ~0.2.

            ``target`` enables early-stop: once a restart passes the gates with
            margin >= target we return immediately. So easy images that flip on
            the deterministic zeros/warm init (r=0) skip the extra restarts; only
            genuinely-hard images pay for all of them.
            """
            eps = level * unit
            alpha = max(eps * cfg.alpha_frac, unit)
            n_restarts = int(restarts if restarts is not None else cfg.restarts)
            best_delta = None
            best_key = float("-inf")
            best_gates, best_margin = False, float("-inf")
            for r in range(max(1, n_restarts)):
                if r == 0 and init is not None:
                    delta = init.detach().clamp(-eps, eps).clone()
                elif r == 0:
                    delta = torch.zeros_like(data1)
                elif cfg.aitl_restarts:
                    delta = _aitl_init(data1, label1, eps)
                else:
                    delta = torch.empty_like(data1).uniform_(-eps, eps)
                delta = _mask(torch.min(torch.max(delta, -data1), 1.0 - data1))
                momentum = torch.zeros_like(data1)
                adam_m = torch.zeros_like(data1)
                adam_v = torch.zeros_like(data1)
                for t in range(1, int(cfg.inner_iters) + 1):
                    if cfg.dual_example:
                        # DualMIFGSM / Ens-FGSM-MIFGSM: gradient averaged over
                        # `dual_ensemble` independent random δ in the ε-ball.
                        # Separates gradient estimation (exploration) from the
                        # momentum-tracked δ (exploitation).
                        g = torch.zeros_like(delta)
                        for _ in range(max(1, cfg.dual_ensemble)):
                            d_rand = _mask(torch.empty_like(data1).uniform_(-eps, eps))
                            d_rand = _mask(torch.min(torch.max(d_rand, -data1), 1.0 - data1))
                            g = g + _smooth(_grad(data1, label1, d_rand))
                        g = g / max(1, cfg.dual_ensemble)
                        cont_margin = None
                    else:
                        g, cont_margin = _grad_m(data1, label1, delta)
                        g = _smooth(g)
                    # Early-stop: _grad_m already scored the current delta, so a
                    # positive continuous gap gates the quantized flip confirm
                    # (no extra forward until the sample is actually flipping).
                    if (cont_margin is not None and cont_margin > 0.0
                            and _flipped(model, data1, label1, delta)):
                        break
                    if cfg.use_adam:
                        # Sign-of-Adam: bias-corrected Adam direction, then sign step.
                        adam_m = cfg.adam_beta1 * adam_m + (1.0 - cfg.adam_beta1) * g
                        adam_v = cfg.adam_beta2 * adam_v + (1.0 - cfg.adam_beta2) * g * g
                        m_hat = adam_m / (1.0 - cfg.adam_beta1 ** t)
                        v_hat = adam_v / (1.0 - cfg.adam_beta2 ** t)
                        direction = m_hat / (v_hat.sqrt() + cfg.adam_eps)
                        delta = _mask(delta + alpha * direction.sign())
                    else:
                        momentum = cfg.decay * momentum + g / (g.abs().mean() + 1e-12)
                        delta = _mask(delta + alpha * momentum.sign())
                    delta = delta.clamp(-eps, eps)
                    delta = _mask(torch.min(torch.max(delta, -data1), 1.0 - data1))
                gates, _, margin = _eval(model, data1, label1, delta)
                key = margin + (1000.0 if gates else 0.0)
                if best_delta is None or key > best_key:
                    best_key = key
                    best_delta = delta.detach().clone()
                    best_gates, best_margin = gates, margin
                if target is not None and gates and margin >= target:
                    break
            return best_delta, best_gates, best_margin

        def _sparse_pgd(data1, label1, eps, k, init, steps, restarts=1):
            """Iterative hard-thresholding PGD: keep the top-k gradient-salient
            elements at +/-eps and re-optimise their sign pattern each step.

            Unlike prune-from-dense, the surviving pixels are RE-OPTIMISED, so a
            far smaller support can still rebuild the flip margin. L-inf stays at
            eps (== chosen level), so RMSE ~ sqrt(k/N)*eps shrinks with k.

            ``restarts`` > 1 reseeds the support from random deltas (after the
            warm start), so an unlucky top-k pick at a small k gets extra chances
            to converge -- letting the k search settle lower. Returns the best
            (delta, ok, margin) over restarts.
            """
            numel = init.numel()
            best_delta = None
            best_key = float("-inf")
            best_ok, best_margin = False, float("-inf")
            for r in range(max(1, int(restarts))):
                if r == 0:
                    delta = init.detach().clone()
                else:
                    delta = _mask(torch.empty_like(data1).uniform_(-eps, eps))
                    delta = torch.min(torch.max(delta, -data1), 1.0 - data1)
                momentum = torch.zeros_like(data1)
                adam_m = torch.zeros_like(data1)
                adam_v = torch.zeros_like(data1)
                for t in range(1, int(steps) + 1):
                    if cfg.dual_example:
                        g = torch.zeros_like(delta)
                        for _ in range(max(1, cfg.dual_ensemble)):
                            d_rand = _mask(torch.empty_like(data1).uniform_(-eps, eps))
                            d_rand = _mask(torch.min(torch.max(d_rand, -data1), 1.0 - data1))
                            g = g + _smooth(_grad(data1, label1, d_rand))
                        g = g / max(1, cfg.dual_ensemble)
                    else:
                        g = _smooth(_grad(data1, label1, delta))
                    if cfg.use_adam:
                        adam_m = cfg.adam_beta1 * adam_m + (1.0 - cfg.adam_beta1) * g
                        adam_v = cfg.adam_beta2 * adam_v + (1.0 - cfg.adam_beta2) * g * g
                        m_hat = adam_m / (1.0 - cfg.adam_beta1 ** t)
                        v_hat = adam_v / (1.0 - cfg.adam_beta2 ** t)
                        direction = m_hat / (v_hat.sqrt() + cfg.adam_eps)
                    else:
                        momentum = cfg.decay * momentum + g / (g.abs().mean() + 1e-12)
                        direction = momentum
                    flat = (direction.abs() * (alive if alive is not None else 1.0)).flatten()
                    if k < numel:
                        keep = torch.topk(flat, k).indices
                        mask = torch.zeros_like(flat, dtype=torch.bool)
                        mask[keep] = True
                        mask = mask.view_as(direction)
                    else:
                        mask = torch.ones_like(direction, dtype=torch.bool)
                    if alive is not None:
                        mask = mask & alive.bool()
                    delta = _mask(mask * (eps * direction.sign()))
                    delta = torch.min(torch.max(delta, -data1), 1.0 - data1)
                gates, _, margin = _eval(model, data1, label1, delta)
                ok = gates and margin >= cfg.flip_margin
                # Prefer passing solutions, then FEWER active pixels (lower RMSE
                # at fixed eps), then larger margin. The tiny margin weight only
                # breaks exact pixel-count ties, so restarts never trade RMSE for
                # surplus margin -- removing the variance seen in the prior run.
                active = int((delta.abs() > 0).sum().item())
                key = (1.0e9 if ok else 0.0) - active + 1e-4 * margin
                if best_delta is None or key > best_key:
                    best_key = key
                    best_delta = delta.detach().clone()
                    best_ok, best_margin = ok, margin
            return best_delta.detach(), best_ok, best_margin

        def _polish(data1, label1, delta1, eps):
            """Greedy backward elimination after the k search.

            The top-k support PGD returns keeps the k most salient pixels, but
            some are redundant once the others lock in the flip. Here we rank the
            active pixels by their current contribution to the margin
            (grad * delta; small/negative == least useful) and try to ZERO the
            least-useful ones in batches, re-verifying the gates + flip_margin
            after every removal. Removal-only keeps L-inf at eps, so every
            accepted drop strictly lowers RMSE. After each round we re-optimise
            the surviving support so later rounds can remove still more.
            """
            cur = delta1.detach().clone()
            gates, _, margin = _eval(model, data1, label1, cur)
            if not (gates and margin >= cfg.flip_margin):
                return cur  # nothing safe to trim
            for _ in range(int(cfg.polish_rounds)):
                g = _grad(data1, label1, cur)
                contrib = (g * cur).flatten()
                active = cur.abs().flatten() > 0
                n_active = int(active.sum().item())
                if n_active <= 1:
                    break
                # least-useful active pixels first; inactive pushed to the end.
                ranked = torch.where(
                    active, contrib, torch.full_like(contrib, float("inf"))
                )
                order = torch.argsort(ranked)[:n_active]
                ptr = 0
                batch = max(1, n_active // 2)
                removed_any = False
                while ptr < n_active and batch >= 1:
                    take = order[ptr : ptr + batch]
                    trial = cur.clone()
                    tf = trial.flatten()
                    tf[take] = 0.0
                    trial = tf.view_as(cur)
                    # Removal-only keeps L-inf == eps and only improves SSIM/PSNR,
                    # so those gates can't newly fail — margin is the only check.
                    margin = _margin(model, data1, trial)
                    if margin >= cfg.flip_margin:
                        cur = trial
                        ptr += batch
                        removed_any = True
                    else:
                        batch //= 2
                if not removed_any:
                    break
                k = int((cur.abs().flatten() > 0).sum().item())
                # Re-optimise the surviving support from the warm start only
                # (restarts=1): polish already has a good support, so random
                # reseeds here add downside risk without RMSE benefit.
                cur2, ok, _ = _sparse_pgd(
                    data1, label1, eps, k, cur, cfg.sparse_steps, restarts=1
                )
                if ok:
                    cur = cur2
            return cur.detach()

        def _sparse_one(data1, label1, delta1, eps):
            """Binary-search the minimum number of perturbed elements (k) for
            which support-restricted PGD still passes every gate + flip_margin,
            then greedily polish away any remaining redundant pixels."""
            di = delta1[0]
            n_active = int(((di.abs() * 255.0).round() != 0).sum().item())
            if n_active <= 1:
                return delta1
            best = delta1.detach().clone()
            warm = delta1
            lo, hi = 1, n_active
            while lo <= hi:
                mid = (lo + hi) // 2
                d, ok, _ = _sparse_pgd(
                    data1, label1, eps, mid, warm, cfg.sparse_steps, cfg.sparse_restarts
                )
                if ok:
                    best, warm = d, d
                    hi = mid - 1
                else:
                    lo = mid + 1
            return _polish(data1, label1, best, eps)

        outs = []
        for i in range(int(data.shape[0])):
            data1 = data[i : i + 1]
            label1 = label[i : i + 1]
            # Cache the true index + clean SSIM stats once for this image so the
            # inner loops avoid a per-call label sync and per-eval mu_x/sigma_x.
            _set_image(data1, label1)
            alive = (
                compute_preprocess_alive_mask(data1)
                if cfg.crop_mask
                else None
            )
            # Reset L2T aug_param fresh for each new image.
            if cfg.grad_diversity > 0 and cfg.grad_transform_src == "l2t":
                _l2t_aug_param = torch.zeros(N_L2T_OPS, device=device)

            # Ascending, SCORE-AWARE level search. The L-inf term carries 0.7
            # weight, so a lower level almost always wins even if it costs all
            # sparsity (1/255 dense ~0.88 >> 2/255 sparse ~0.76). We therefore
            # accept the smallest level that flips with only `accept_margin`
            # (conservative-but-lower); the heavier `flip_margin` is reserved for
            # pruning, where thin margins are fragile. If the accepted level's
            # dense margin is below flip_margin, the sparse step simply returns
            # the dense delta (still 1/255 -- the big win), instead of falling
            # back to a higher level.
            chosen = None
            chosen_ok = False
            chosen_level = cfg.max_level
            warm = None
            for level in range(1, cfg.max_level + 1):
                # Level 1 is 0.7-weighted and worth ~0.2 over level 2, so spend
                # extra restarts there to reliably flip borderline-hard images.
                # Early-stop once a restart clears flip_margin so easy images
                # don't pay for the extra restarts (only hard ones do).
                n_restarts = cfg.first_level_restarts if level == 1 else cfg.restarts
                d, gates, margin = _pgd_at_level(
                    data1, label1, level,
                    init=warm, restarts=n_restarts, target=cfg.flip_margin,
                )
                warm = d
                if gates and margin >= cfg.accept_margin:
                    chosen, chosen_ok, chosen_level = d, True, level
                    break
            if chosen is None:
                chosen = warm  # best effort (nothing flipped within budget)

            if cfg.sparse and chosen_ok:
                chosen = _sparse_one(data1, label1, chosen, chosen_level * unit)
            outs.append(chosen.detach())

        return torch.cat(outs, dim=0)

    return budget_forward


def build_ops_budget_attack(
    device: torch.device,
    *,
    num_sample_neighbor: int = 0,
    num_sample_operator: int = 0,
    beta: float = 2.0,
    decay: float = 1.0,
    budget: BudgetConfig | None = None,
    surrogate: EfficientNetV2LSurrogate | None = None,
):
    """OPS pinned to EfficientNetV2-L with a minimum-norm (L-inf/SSIM-aware) loop.

    Reuses OPS's averaged-gradient flip direction (operator/neighbor sampling is
    optional; default off == fast white-box overfit) but replaces the plain
    CE-ascent forward with a budget-minimizing objective that targets the
    validator reward directly.

    Returns an OPS instance whose ``__call__(data, label)`` returns the
    minimum-norm delta; ``attack.model`` is the validator-exact classifier head.
    """
    OPS = _import_ops_class()
    cfg = budget or BudgetConfig()
    head = surrogate if surrogate is not None else EfficientNetV2LSurrogate(device)
    budget_forward = _make_budget_forward(cfg)

    class OPSBudgetMinEffNet(OPS):
        """OPS + minimum-norm objective, bound to EfficientNetV2-L."""

        def load_model(self, model_name):  # noqa: ARG002 - name ignored on purpose
            return head

        def forward(self, data, label, **kwargs):
            return budget_forward(self, data, label, **kwargs)

    attack = OPSBudgetMinEffNet(
        model_name="efficientnet_v2_l",
        epsilon=float(cfg.max_level) / 255.0,
        beta=float(beta),
        num_iter=int(cfg.inner_iters),
        num_sample_neighbor=int(num_sample_neighbor),
        num_sample_operator=int(num_sample_operator),
        decay=float(decay),
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=device,
    )
    return attack
