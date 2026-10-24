"""Decode, bound and re-encode outbound images without metadata."""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageOps

from .policy import PolicyError

MAX_DIMENSION = 2048
MAX_BYTES = 4 * 1024 * 1024


def normalize_image_bytes(data: bytes) -> tuple[bytes, str, tuple[int, int]]:
    if not data:
        raise PolicyError("empty_image")
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source)
            image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            has_alpha = image.mode == "RGBA"
            out = io.BytesIO()
            if has_alpha:
                image.save(out, format="PNG", optimize=True)
                mime = "image/png"
            else:
                image.save(out, format="JPEG", quality=88, optimize=True)
                mime = "image/jpeg"
            payload = out.getvalue()
            dimensions = image.size
    except PolicyError:
        raise
    except Exception as exc:
        raise PolicyError("invalid_image") from exc
    if len(payload) > MAX_BYTES:
        raise PolicyError("normalized_image_too_large")
    return payload, mime, dimensions


def normalize_data_url(url: str) -> tuple[str, int, tuple[int, int]]:
    if not url.startswith("data:") or ";base64," not in url:
        raise PolicyError("remote_image_urls_forbidden")
    try:
        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
    except Exception as exc:
        raise PolicyError("invalid_image_base64") from exc
    payload, mime, dimensions = normalize_image_bytes(raw)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}", len(payload), dimensions


def normalize_message_images(messages: list[dict]) -> tuple[list[dict], dict[str, int]]:
    import copy

    result = copy.deepcopy(messages)
    count = 0
    total = 0
    for message in result:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url") or {}
            normalized, size, _dimensions = normalize_data_url(str(image_url.get("url", "")))
            part["image_url"] = {"url": normalized}
            count += 1
            total += size
    return result, {"image_count": count, "image_bytes": total}
