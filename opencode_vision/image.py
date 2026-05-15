"""
Image processing utilities.

Handles reading, resizing, and encoding images for both
local OCR engines and the Gemini Vision API.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ── Optional Pillow ─────────────────────────────────────────────────────────

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── MIME type mapping ───────────────────────────────────────────────────────

MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".avif": "image/avif",
    ".heic": "image/heic",
    ".heif": "image/heif",
}

MAX_PIXELS = 2048 * 2048  # resize if larger


def _resize_if_needed(img) -> tuple:
    """Return (resized_image, was_resized)."""
    if img.width * img.height <= MAX_PIXELS:
        return img, False
    ratio = (MAX_PIXELS / (img.width * img.height)) ** 0.5
    new_w = int(img.width * ratio)
    new_h = int(img.height * ratio)
    log.info("Resized %dx%d → %dx%d", img.width, img.height, new_w, new_h)
    return img.resize((new_w, new_h), Image.LANCZOS), True


def to_base64(image_path: str) -> tuple:
    """Read image, optionally resize → (b64_data, error_message, mime_type).

    Args:
        image_path: Path to image file (supports ~ expansion).

    Returns:
        (b64_string, None, mime_type) on success.
        (None, error_message, None) on failure.
    """
    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        return None, f"File not found: {image_path}", None
    if not path.is_file():
        return None, f"Not a file: {image_path}", None

    ext = path.suffix.lower()
    mime = MIME_MAP.get(ext, "image/png")

    try:
        if HAS_PIL:
            img = Image.open(path)
            img, _ = _resize_if_needed(img)
            buf = io.BytesIO()
            save_fmt = "PNG" if ext not in (".jpg", ".jpeg") else "JPEG"
            img.save(buf, format=save_fmt)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64, None, mime
    except Exception as e:
        log.error("Image processing error: %s", e)
        return None, f"Image processing error: {e}", None


def get_metadata(image_path: str) -> dict:
    """Extract image metadata (format, dimensions, size).

    Returns a dict with available metadata. All fields optional.
    """
    meta = {}
    path = Path(image_path).expanduser().resolve()

    if HAS_PIL:
        try:
            img = Image.open(path)
            meta["format"] = img.format or "unknown"
            meta["width"] = img.width
            meta["height"] = img.height
            meta["mode"] = img.mode
            if img.height:
                meta["aspect_ratio"] = round(img.width / img.height, 2)
        except Exception:
            pass

    try:
        kb = path.stat().st_size / 1024
        meta["file_size_kb"] = round(kb, 1)
    except Exception:
        pass

    return meta
