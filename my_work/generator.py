"""Perturbation generator architectures."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with a group count that divides `channels` (batch/res independent)."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _init_head(head: nn.Conv2d, max_linf: float, head_init: float) -> None:
    """Initialise a ``tanh(head) * max_linf`` perturbation head.

    ``head_init <= 0`` keeps the original zero-init (perturbation ~0 at start —
    stable but starts inside the uint8-quantization dead zone). ``head_init > 0``
    seeds the bias so the initial L-inf perturbation is ~``head_init`` (with a
    random per-channel sign), which lifts the generator above the quantization
    floor from step 1 so the classifier sees a real change immediately. The
    weights stay zero so the input-conditioned structure is learned from the
    first gradient step.
    """
    nn.init.zeros_(head.weight)
    if head.bias is None:
        return
    if head_init <= 0.0:
        nn.init.zeros_(head.bias)
        return
    # tanh(b) * max_linf == head_init  ->  b == atanh(head_init / max_linf)
    ratio = min(max(float(head_init) / max(float(max_linf), 1e-12), 0.0), 0.999)
    bias_val = math.atanh(ratio)
    with torch.no_grad():
        signs = torch.where(
            torch.rand_like(head.bias) < 0.5, -1.0, 1.0
        )
        head.bias.copy_(signs * bias_val)


class ResBlock(nn.Module):
    """Pre-activation residual block. `dilation` widens the receptive field cheaply."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class Down(nn.Module):
    """Strided downsample (x2) followed by a residual block."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)
        self.norm = _group_norm(c_out)
        self.act = nn.SiLU(inplace=True)
        self.res = ResBlock(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.act(self.norm(self.down(x))))


class Up(nn.Module):
    """Upsample to the skip's exact size, concat, fuse, residual block.

    Size-matched interpolation keeps the network valid for arbitrary H, W
    (resolution buckets need not be divisible by the downsample factor).
    """

    def __init__(self, c_in: int, c_skip: int, c_out: int) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(c_in + c_skip, c_out, 3, padding=1)
        self.norm = _group_norm(c_out)
        self.act = nn.SiLU(inplace=True)
        self.res = ResBlock(c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bicubic")
        x = torch.cat([x, skip], dim=1)
        return self.res(self.act(self.norm(self.fuse(x))))


class Generator(nn.Module):
    """
    Multi-scale U-Net perturbation generator.

    Encoder downsamples 3x (large receptive field), a dilated bottleneck widens
    context further, and a skip-connected decoder restores full resolution.
    Output: Tanh x max_linf  ->  L-inf <= max_linf guaranteed by architecture.

    The head is zero-initialized so the initial perturbation is ~0 (adv ~ clean),
    which gives a stable optimisation start before the flip term takes over.
    """

    def __init__(self, max_linf: float = 0.03, base: int = 48, head_init: float = 0.0) -> None:
        super().__init__()
        self.max_linf = float(max_linf)
        b = int(base)

        # Stem (full resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(3, b, 3, padding=1),
            _group_norm(b),
            nn.SiLU(inplace=True),
            ResBlock(b),
        )

        # Encoder
        self.down1 = Down(b, b * 2)        # H/2
        self.down2 = Down(b * 2, b * 4)    # H/4
        self.down3 = Down(b * 4, b * 4)    # H/8

        # Bottleneck — dilations enlarge the receptive field over global structure
        self.mid = nn.Sequential(
            ResBlock(b * 4, dilation=2),
            ResBlock(b * 4, dilation=4),
            ResBlock(b * 4, dilation=8),
        )

        # Decoder (skip connections from stem / down1 / down2)
        self.up3 = Up(b * 4, b * 4, b * 4)  # -> H/4
        self.up2 = Up(b * 4, b * 2, b * 2)  # -> H/2
        self.up1 = Up(b * 2, b, b)          # -> H

        self.head = nn.Conv2d(b, 3, 3, padding=1)
        _init_head(self.head, self.max_linf, head_init)

        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)       # b,    H
        s1 = self.down1(s0)     # 2b,   H/2
        s2 = self.down2(s1)     # 4b,   H/4
        s3 = self.down3(s2)     # 4b,   H/8

        m = self.mid(s3)        # 4b,   H/8

        d3 = self.up3(m, s2)    # 4b,   H/4
        d2 = self.up2(d3, s1)   # 2b,   H/2
        d1 = self.up1(d2, s0)   # b,    H

        return self.tanh(self.head(d1)) * self.max_linf


# ─── LTP ResNet generator (Nakka & Salzmann, NeurIPS 2021) ─────────────────────


class _LTPResidualBlock(nn.Module):
    """ResNet block from Transferable_Perturbations/pix2pix/models/resnet_gen.py."""

    def __init__(self, num_filters: int, gen_dropout: float) -> None:
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
        ]
        if gen_dropout > 0.0:
            layers.append(nn.Dropout(gen_dropout))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, bias=False),
            nn.BatchNorm2d(num_filters),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GeneratorResnet(nn.Module):
    """LTP perturbation generator (faithful port of ``GeneratorResnet``).

    Architecture matches the paper's release: ReflectionPad + 7x7 stem, two
    strided downsamples, ``2`` (low) or ``6`` (high) residual blocks, two
    transposed-conv upsamples, and a 7x7 head.

    Two adaptations make it drop into this subnet pipeline:
      * The head emits a *bounded perturbation* ``tanh(x) * max_linf`` (the
        paper emitted a full image in ``[0, 1]`` then projected to an eps-ball).
        This matches :class:`Generator` so ``forward_adv`` is unchanged.
      * The output is resized back to the input resolution, so the net works on
        the arbitrary native sizes this pipeline uses (the paper assumed fixed
        224/299 inputs). Bilinear resampling of values in ``[-m, m]`` stays in
        ``[-m, m]``, so the L-inf bound is preserved.

    The head conv is zero-initialised (perturbation ~0 at start) for a stable
    minimum-norm optimisation start, consistent with :class:`Generator`.
    """

    def __init__(
        self,
        max_linf: float = 0.03,
        base: int = 64,
        *,
        gen_dropout: float = 0.0,
        data_dim: str = "high",
        head_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_linf = float(max_linf)
        ngf = int(base)
        self.data_dim = str(data_dim)

        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        n_res = 6 if self.data_dim == "high" else 2
        self.resblocks = nn.Sequential(
            *[_LTPResidualBlock(ngf * 4, gen_dropout) for _ in range(n_res)]
        )

        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
        )

        _init_head(self.blockf[1], self.max_linf, head_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.resblocks(h)
        h = self.upsampl1(h)
        h = self.upsampl2(h)
        h = self.blockf(h)

        pert = torch.tanh(h) * self.max_linf
        if pert.shape[-2:] != x.shape[-2:]:
            pert = F.interpolate(pert, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return pert


def build_generator(
    arch: str = "unet",
    *,
    max_linf: float = 0.03,
    base: int = 48,
    gen_dropout: float = 0.0,
    head_init: float = 0.0,
) -> nn.Module:
    """Factory for the selectable generator architectures.

    ``unet``   -> :class:`Generator` (default; resolution-agnostic U-Net).
    ``resnet`` -> :class:`GeneratorResnet` (LTP port).

    ``head_init`` seeds the perturbation head so the initial L-inf is ~that value
    (default 0 = zero-init). A small positive value (e.g. 0.004) lifts the
    generator above the uint8-quantization dead zone at the start of training.
    """
    arch = str(arch).lower()
    if arch == "unet":
        return Generator(max_linf=max_linf, base=base, head_init=head_init)
    if arch == "resnet":
        return GeneratorResnet(
            max_linf=max_linf, base=base, gen_dropout=gen_dropout, head_init=head_init
        )
    raise ValueError(f"unknown gen-arch: {arch!r} (expected 'unet' or 'resnet')")


def load_generator_checkpoint(
    generator: Generator,
    path: Path | str,
    *,
    strict: bool = True,
) -> dict:
    """Load weights from a training checkpoint or raw state_dict.

    Accepts files saved by train_generator*.py:
      {"epoch", "max_linf", "generator_state", "val", ...}
    or a bare state_dict mapping param names -> tensors.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "generator_state" in payload:
        state = payload["generator_state"]
        meta = {
            k: payload[k]
            for k in ("epoch", "max_linf", "val", "gen_arch", "train_score")
            if k in payload
        }
    elif isinstance(payload, dict):
        state = payload
        meta = {}
    else:
        raise ValueError(f"unsupported checkpoint format: {path}")

    incompatible = generator.load_state_dict(state, strict=strict)
    if not strict:
        print(
            f"checkpoint load (strict=False): "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    meta["path"] = str(path)
    return meta
