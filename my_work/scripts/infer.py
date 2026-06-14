#!/usr/bin/env python3
"""Run EfficientNet-V2-L inference exactly like the Perturb validator."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import torch

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from perturb_mirror.image_io import decode_image_b64
from perturb_mirror.model import (
    _preprocess_for_efficientnet_v2_l,
    load_efficientnet_v2_l,
    normalize_prediction_label,
    predict_index,
    predict_label,
    PREPROCESS,
)


def infer_from_path(image_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device)

    # Reproduce the validator's exact entry point: file/bytes -> base64 -> decode.
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # validator.py:
    #   image = decode_image_b64(image_b64).to(self.device)
    #   predicted = predict_label(self.model, image)
    image = decode_image_b64(image_b64).to(device)

    print(f"decoded tensor      : shape={tuple(image.shape)} dtype={image.dtype} "
          f"min={image.min():.4f} max={image.max():.4f}")
    preprocessed = _preprocess_for_efficientnet_v2_l(image.unsqueeze(0))
    print(f"preprocessed tensor : shape={tuple(preprocessed.shape)} dtype={preprocessed.dtype} "
          f"min={preprocessed.min():.4f} max={preprocessed.max():.4f}")
    print(f"PREPROCESS object   : {PREPROCESS}")

    idx = predict_index(model, image)
    label = predict_label(model, image)
    print(f"predicted index     : {idx}")
    print(f"predicted label     : {label!r}")
    print(f"normalized label    : {normalize_prediction_label(label)!r}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/infer.py <image_path>")
        return 2
    infer_from_path(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
