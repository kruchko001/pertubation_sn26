#!/usr/bin/env python3
"""Download one Pexels image per Perturb validator prompt for local inspection."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from paths import PERTURB, PROMPT_SAMPLES_DIR, ROOT

if str(PERTURB) not in sys.path:
    sys.path.insert(0, str(PERTURB))

from perturbnet.constants import IMAGE_ENDPOINT, PEXELS_IMAGE_VARIANT, PROMPTS


def _slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.strip().lower()).strip("_")


def _resolve_api_key(cli_key: str) -> str:
    return (
        cli_key.strip()
        or os.getenv("PERTURB_PEXELS_API_KEY", "").strip()
        or os.getenv("PEXELS_API_KEY", "").strip()
    )


def _pick_image_url(photo: dict, variant: str) -> str | None:
    src = photo.get("src", {}) if isinstance(photo, dict) else {}
    if not isinstance(src, dict):
        return None
    url = (
        src.get(variant)
        or src.get("medium")
        or src.get("large")
        or src.get("large2x")
        or src.get("original")
    )
    return url if isinstance(url, str) and url.strip() else None


def _extension_from_url(url: str, content_type: str) -> str:
    lowered = url.lower().split("?", 1)[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if lowered.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    return ".jpg"


def fetch_one(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    prompt: str,
    variant: str,
) -> tuple[bytes, dict]:
    response = session.get(
        endpoint,
        params={"query": prompt, "page": 1, "per_page": 1},
        headers={"Authorization": api_key},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    photos = data.get("photos") if isinstance(data, dict) else None
    if not isinstance(photos, list) or not photos:
        raise RuntimeError(f"no photos returned for prompt={prompt!r}")
    photo = photos[0]
    image_url = _pick_image_url(photo, variant)
    if not image_url:
        raise RuntimeError(f"photo has no usable src url for prompt={prompt!r}")

    image_response = session.get(image_url, timeout=30)
    image_response.raise_for_status()
    if not image_response.content:
        raise RuntimeError(f"empty image body for prompt={prompt!r}")

    meta = {
        "prompt": prompt,
        "pexels_id": photo.get("id"),
        "photographer": photo.get("photographer"),
        "photographer_url": photo.get("photographer_url"),
        "pexels_url": photo.get("url"),
        "image_url": image_url,
    }
    ext = _extension_from_url(image_url, image_response.headers.get("Content-Type", ""))
    meta["extension"] = ext
    return image_response.content, meta


def write_gallery(out_dir: Path, entries: list[dict]) -> None:
    html_path = out_dir / "index.html"
    rows = []
    for entry in sorted(entries, key=lambda item: item["prompt"].lower()):
        filename = entry["filename"]
        prompt = entry["prompt"]
        credit = entry.get("photographer") or "Unknown"
        pexels_url = entry.get("pexels_url") or "#"
        rows.append(
            f"""
        <figure>
          <img src="{filename}" alt="{prompt}" loading="lazy" />
          <figcaption><strong>{prompt}</strong><br />
            Photo by <a href="{pexels_url}">{credit}</a> on Pexels
          </figcaption>
        </figure>"""
        )

    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Perturb prompt samples ({len(entries)} images)</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #aaa; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 20px;
      margin-top: 24px;
    }}
    figure {{
      margin: 0;
      background: #1b1b1b;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid #333;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      background: #222;
    }}
    figcaption {{
      padding: 10px 12px 14px;
      font-size: 14px;
      line-height: 1.4;
    }}
    a {{ color: #7cb8ff; }}
  </style>
</head>
<body>
  <h1>Perturb validator prompts</h1>
  <p>One Pexels image per prompt ({len(entries)} total). Open this file in a browser to browse.</p>
  <div class="grid">
    {"".join(rows)}
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one Pexels image per Perturb prompt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROMPT_SAMPLES_DIR,
        help="Directory for downloaded images",
    )
    parser.add_argument("--pexels-api-key", default="", help="Pexels API key (or set PERTURB_PEXELS_API_KEY)")
    parser.add_argument("--image-endpoint", default=IMAGE_ENDPOINT)
    parser.add_argument("--pexels-image-variant", default=PEXELS_IMAGE_VARIANT)
    parser.add_argument("--delay-seconds", type=float, default=0.25, help="Pause between API calls")
    parser.add_argument("--limit", type=int, default=0, help="Download only first N prompts (0 = all)")
    args = parser.parse_args()

    api_key = _resolve_api_key(args.pexels_api_key)
    if not api_key:
        print(
            "Missing Pexels API key. Set PERTURB_PEXELS_API_KEY or pass --pexels-api-key.\n"
            "Free key: https://www.pexels.com/api/",
            file=sys.stderr,
        )
        return 1

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = list(PROMPTS)
    if args.limit > 0:
        prompts = prompts[: args.limit]

    session = requests.Session()
    entries: list[dict] = []
    failures: list[dict] = []

    print(f"Downloading {len(prompts)} images to {out_dir}")
    for index, prompt in enumerate(prompts, start=1):
        slug = _slug(prompt)
        print(f"[{index}/{len(prompts)}] {prompt!r} -> {slug}", flush=True)
        try:
            content, meta = fetch_one(
                session=session,
                endpoint=args.image_endpoint,
                api_key=api_key,
                prompt=prompt,
                variant=args.pexels_image_variant,
            )
            filename = f"{slug}{meta['extension']}"
            path = out_dir / filename
            path.write_bytes(content)
            entry = {
                "prompt": prompt,
                "filename": filename,
                "path": str(path.relative_to(ROOT)),
                **meta,
            }
            entries.append(entry)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures.append({"prompt": prompt, "error": str(exc)})
        if index < len(prompts) and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    manifest = {
        "count_ok": len(entries),
        "count_failed": len(failures),
        "output_dir": str(out_dir),
        "entries": entries,
        "failures": failures,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_gallery(out_dir, entries)

    print(f"\nDone: {len(entries)} ok, {len(failures)} failed")
    print(f"Gallery: {out_dir / 'index.html'}")
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
