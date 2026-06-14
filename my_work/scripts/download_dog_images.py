#!/usr/bin/env python3
"""Download all dog images from Pexels for the Perturb 'dog' prompt."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from paths import DOG_IMAGES_DIR

PROMPT = "dog"
ENDPOINT = "https://api.pexels.com/v1/search"
PER_PAGE = 80  # Pexels max
DELAY_SECONDS = 0.3


def _api_key() -> str:
    return (
        os.getenv("PERTURB_PEXELS_API_KEY", "").strip()
        or os.getenv("PEXELS_API_KEY", "").strip()
    )


def _extension(url: str, content_type: str) -> str:
    lowered = url.lower().split("?", 1)[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if lowered.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    return ".jpg"


def main() -> int:
    api_key = _api_key()
    if not api_key:
        print("Missing PERTURB_PEXELS_API_KEY", file=sys.stderr)
        return 1

    output_dir = DOG_IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    headers = {"Authorization": api_key}

    # Discover total pages
    first = session.get(
        ENDPOINT,
        params={"query": PROMPT, "page": 1, "per_page": PER_PAGE},
        headers=headers,
        timeout=30,
    )
    first.raise_for_status()
    data = first.json()
    total_results = int(data.get("total_results", 0))
    print(f"Pexels reports {total_results:,} images for query={PROMPT!r}")

    entries: list[dict] = []
    seen_ids: set[int] = set()
    page = 1
    failures = 0

    while True:
        print(f"Fetching search page {page}...", flush=True)
        if page == 1:
            payload = data
        else:
            response = session.get(
                ENDPOINT,
                params={"query": PROMPT, "page": page, "per_page": PER_PAGE},
                headers=headers,
                timeout=30,
            )
            if response.status_code == 404:
                print(f"  Page {page} not available (404). Stopping pagination.")
                break
            response.raise_for_status()
            payload = response.json()

        photos = payload.get("photos") if isinstance(payload, dict) else None
        if not isinstance(photos, list) or not photos:
            print(f"  No photos on page {page}. Done.")
            break

        for photo in photos:
            if not isinstance(photo, dict):
                continue
            photo_id = photo.get("id")
            if photo_id in seen_ids:
                continue
            seen_ids.add(photo_id)

            src = photo.get("src", {})
            if not isinstance(src, dict):
                continue
            image_url = src.get("original") or src.get("large2x") or src.get("large") or src.get("medium")
            if not isinstance(image_url, str) or not image_url.strip():
                failures += 1
                continue

            try:
                img_resp = session.get(image_url, timeout=60)
                img_resp.raise_for_status()
                if not img_resp.content:
                    failures += 1
                    continue
                ext = _extension(image_url, img_resp.headers.get("Content-Type", ""))
                filename = f"dog_{photo_id}{ext}"
                path = output_dir / filename
                path.write_bytes(img_resp.content)
                entries.append(
                    {
                        "id": photo_id,
                        "filename": filename,
                        "photographer": photo.get("photographer"),
                        "photographer_url": photo.get("photographer_url"),
                        "pexels_url": photo.get("url"),
                        "image_url": image_url,
                        "width": photo.get("width"),
                        "height": photo.get("height"),
                    }
                )
                if len(entries) % 50 == 0:
                    print(f"  Downloaded {len(entries)} images...", flush=True)
            except Exception as exc:
                failures += 1
                print(f"  Failed id={photo_id}: {exc}", file=sys.stderr)

        page += 1
        time.sleep(DELAY_SECONDS)

    manifest = {
        "prompt": PROMPT,
        "pexels_total_reported": total_results,
        "downloaded_count": len(entries),
        "failures": failures,
        "output_dir": str(output_dir),
        "entries": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone: {len(entries)} images saved to {output_dir}")
    print(f"Failures: {failures}")
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
