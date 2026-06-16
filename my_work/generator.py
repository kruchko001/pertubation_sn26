"""Perturbation generator architectures."""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Small U-Net style conv stack.
    Output: Tanh × max_linf  →  L∞ ≤ max_linf guaranteed by architecture.
    """

    def __init__(self, max_linf: float = 0.03) -> None:
        super().__init__()
        self.max_linf = float(max_linf)

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))

        # Decoder (skip connections from encoder)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 32, 3, 3, padding=1))

        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(torch.cat([b, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.tanh(d1) * self.max_linf
