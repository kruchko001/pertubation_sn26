#!/usr/bin/env python3
"""Download all Pexels images for Perturb validator prompt(s)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

MY_WORK = Path(__file__).resolve().parent.parent
PERTURB = MY_WORK.parent / "Perturb"
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))
if str(PERTURB) not in sys.path:
    sys.path.insert(0, str(PERTURB))

from paths import PROMPT_IMAGES_DIR, prompt_images_dir, slugify_prompt
from perturbnet.constants import IMAGE_ENDPOINT, PROMPTS

ENDPOINT = IMAGE_ENDPOINT
PER_PAGE = 80  # Pexels max
DEFAULT_DELAY_SECONDS = 0.3


def _api_key(cli_key: str) -> str:
    return (
        cli_key.strip()
        or os.getenv("PERTURB_PEXELS_API_KEY", "").strip()
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


def _resolve_prompts(input_prompt: str) -> list[str]:
    value = input_prompt.strip()
    if value.upper() == "ALL":
        return list(PROMPTS)

    lowered = value.lower()
    for prompt in PROMPTS:
        if prompt.lower() == lowered:
            return [prompt]

    known = ", ".join(repr(p) for p in PROMPTS[:5])
    raise ValueError(
        f"Unknown prompt {value!r}. Use ALL or one of the {len(PROMPTS)} validator prompts "
        f"(e.g. {known}, ...)."
    )


def download_prompt(
    session: requests.Session,
    api_key: str,
    prompt: str,
    output_dir: Path,
    *,
    delay_seconds: float,
    image_variant: str,
) -> dict:
    slug = slugify_prompt(prompt)
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": api_key}

    first = session.get(
        ENDPOINT,
        params={"query": prompt, "page": 1, "per_page": PER_PAGE},
        headers=headers,
        timeout=30,
    )
    first.raise_for_status()
    data = first.json()
    total_results = int(data.get("total_results", 0) or 0)
    print(f"\n[{prompt!r}] Pexels reports {total_results:,} images -> {output_dir}")

    entries: list[dict] = []
    seen_ids: set[int] = set()
    page = 1
    failures = 0

    while True:
        print(f"  Fetching page {page}...", flush=True)
        if page == 1:
            payload = data
        else:
            response = session.get(
                ENDPOINT,
                params={"query": prompt, "page": page, "per_page": PER_PAGE},
                headers=headers,
                timeout=30,
            )
            if response.status_code == 404:
                print(f"  Page {page} not available (404). Stopping.")
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
            image_url = (
                src.get(image_variant)
                or src.get("original")
                or src.get("large2x")
                or src.get("large")
                or src.get("medium")
            )
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
                filename = f"{slug}_{photo_id}{ext}"
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
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    manifest = {
        "prompt": prompt,
        "slug": slug,
        "pexels_total_reported": total_results,
        "downloaded_count": len(entries),
        "failures": failures,
        "output_dir": str(output_dir),
        "entries": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Saved {len(entries)} images ({failures} failures)")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download all Pexels images for one or all Perturb validator prompts"
    )
    parser.add_argument(
        "--input_prompt",
        required=True,
        help='Prompt label (e.g. "dog", "sports car") or ALL for every prompt in constants.py',
    )
    parser.add_argument("--pexels-api-key", default="", help="Pexels API key (or set PERTURB_PEXELS_API_KEY)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (single-prompt mode only)",
    )
    parser.add_argument(
        "--image-variant",
        default=os.getenv("PERTURB_PEXELS_IMAGE_VARIANT", "original"),
        help="Pexels src variant: original, large2x, large, medium (default: original)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Pause between Pexels search pages",
    )
    args = parser.parse_args()

    try:
        prompts = _resolve_prompts(args.input_prompt)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    api_key = _api_key(args.pexels_api_key)
    if not api_key:
        print("Missing Pexels API key. Set PERTURB_PEXELS_API_KEY or pass --pexels-api-key.", file=sys.stderr)
        return 1

    if args.output_dir is not None and len(prompts) != 1:
        print("--output-dir is only supported when --input_prompt is a single prompt.", file=sys.stderr)
        return 1

    session = requests.Session()
    results: list[dict] = []

    for index, prompt in enumerate(prompts, start=1):
        if len(prompts) > 1:
            print(f"\n=== Prompt {index}/{len(prompts)} ===")
        out_dir = args.output_dir.resolve() if args.output_dir else prompt_images_dir(prompt)
        manifest = download_prompt(
            session=session,
            api_key=api_key,
            prompt=prompt,
            output_dir=out_dir,
            delay_seconds=args.delay_seconds,
            image_variant=args.image_variant.strip().lower(),
        )
        results.append(manifest)

    if len(prompts) > 1:
        PROMPT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "input_prompt": "ALL",
            "prompt_count": len(prompts),
            "total_downloaded": sum(item["downloaded_count"] for item in results),
            "total_failures": sum(item["failures"] for item in results),
            "prompts": [
                {
                    "prompt": item["prompt"],
                    "slug": item["slug"],
                    "downloaded_count": item["downloaded_count"],
                    "failures": item["failures"],
                    "output_dir": item["output_dir"],
                }
                for item in results
            ],
        }
        summary_path = PROMPT_IMAGES_DIR / "manifest_all.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSummary: {summary_path}")
        print(
            f"Done: {summary['total_downloaded']} images across {len(prompts)} prompts "
            f"({summary['total_failures']} failures)"
        )
    else:
        item = results[0]
        print(f"\nDone: {item['downloaded_count']} images saved to {item['output_dir']}")

    return 0 if sum(item["downloaded_count"] for item in results) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
