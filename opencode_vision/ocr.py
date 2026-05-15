"""
OCR engine — PaddleOCR primary, Gemini Vision fallback.

Architecture:
    ┌─ User image ──┐
    │               │
    ▼               ▼
    ┌─────────────────────────┐    high conf    ┌──────────────┐
    │ PaddleOCR (PP-OCRv5)   │ ──────────────►  │ Return text  │
    │ • Best accuracy (0% err)│    ≥ 85%        └──────────────┘
    │ • 100+ languages        │
    │ • Apache 2.0            │
    └────────┬────────────────┘
             │ low conf / error
             ▼
    ┌─────────────────────────┐
    │ Gemini Vision API       │
    │ • Handwriting / scene   │
    │ • 1,500 free req/day    │
    └────────┬────────────────┘
             │
             ▼
    ┌──────────────┐
    │ Return text  │
    └──────────────┘

Why PaddleOCR:
    - 4.5% CER vs Tesseract's 18.2% (4× more accurate)
    - 0 errors on financial document benchmark (2026)
    - 92.86 OmniDocBench score (SOTA open source)
    - ~15MB total model size, Apache 2.0 license
    - Faster than EasyOCR once warm (12.7 FPS on GPU)

Why Gemini fallback:
    - PaddleOCR struggles with handwriting (14.4% accuracy)
    - Gemini achieves 86%+ on handwritten text
    - Also handles scene text and degraded images well
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from opencode_vision import gemini
from opencode_vision import image as img_util

log = logging.getLogger(__name__)

# ── PaddleOCR (optional, best-in-class local OCR) ──────────────────────────
#
# PaddleOCR PP-OCRv5 is the most accurate open-source OCR engine (92.86
# OmniDocBench, 0% error on invoice benchmarks). However, it requires the
# PaddlePaddle framework which may not be available on all platforms (e.g.,
# Python 3.14+). When unavailable, the engine gracefully falls back to
# Gemini Vision API with zero code changes.
#
# Once PaddleOCR fails to initialize (e.g., missing PaddlePaddle), we
# permanently mark it as unavailable to avoid repeated cold-start attempts.

HAS_PADDLE = False
_PADDLE_DEAD = False  # permanently disabled after first init failure
_ocr_instance = None

try:
    from paddleocr import PaddleOCR as _PaddleOCR  # type: ignore[import-untyped]

    HAS_PADDLE = True
except ImportError:
    pass


def _get_paddle():
    """Lazy-init PaddleOCR singleton. Returns None permanently on failure."""
    global _ocr_instance, _PADDLE_DEAD

    if _PADDLE_DEAD:
        return None
    if _ocr_instance is not None:
        return _ocr_instance
    if not HAS_PADDLE:
        return None

    log.info("Initializing PaddleOCR (PP-OCRv5)...")
    t0 = time.time()
    try:
        _ocr_instance = _PaddleOCR(
            use_textline_orientation=True,  # auto-rotate text (was use_angle_cls)
            lang="en",
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
        )
        elapsed = time.time() - t0
        log.info("PaddleOCR initialized in %.1fs", elapsed)
        return _ocr_instance
    except Exception as e:
        log.warning("PaddleOCR init failed (engine unavailable): %s", e)
        _PADDLE_DEAD = True
        return None


CONTENT_CHECK_WORDS = frozenset({
    # Empty or near-empty results
    "no text", "no text detected", "no readable text",
    "nothing", "empty", "blank",
    # Common tesseract-style failure outputs
    "text", "content", "image",
})


def _is_empty_result(text: str | None) -> bool:
    """Check if OCR result is effectively empty."""
    if not text:
        return True
    cleaned = text.strip().lower()
    if len(cleaned) < 3:
        return True
    if cleaned in CONTENT_CHECK_WORDS:
        return True
    return False


# ── Public API ──────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.70  # PaddleOCR confidence threshold for fallback


def extract_text(image_path: str) -> dict:
    """Extract text using PaddleOCR, fall back to Gemini.

    Args:
        image_path: Absolute path to image file.

    Returns:
        {"text": "extracted content..."} on success.
        {"error": "reason..."} on total failure.
    """
    path = str(Path(image_path).expanduser().resolve())

    # 1. Try PaddleOCR
    if HAS_PADDLE:
        result = _ocr_extract(path)
        if result and not _is_empty_result(result.get("text")):
            conf = result.get("confidence", 0)
            if conf >= CONFIDENCE_THRESHOLD:
                log.info("PaddleOCR succeeded (conf=%.2f)", conf)
                return {"text": result["text"]}
            log.info(
                "PaddleOCR confidence too low (%.2f < %.2f), "
                "falling back to Gemini",
                conf, CONFIDENCE_THRESHOLD,
            )
        elif result and _is_empty_result(result.get("text")):
            log.info("PaddleOCR returned empty result, trying Gemini...")
        else:
            log.info("PaddleOCR failed, trying Gemini...")
    else:
        log.info("PaddleOCR not installed, using Gemini...")

    # 2. Gemini fallback
    b64, err, mime = img_util.to_base64(path)
    if err:
        return {"error": err}
    return gemini.call(gemini.OCR_PROMPT, b64, mime)


def describe_image(image_path: str, prompt: str | None = None) -> dict:
    """Describe an image using Gemini Vision API."""
    b64, err, mime = img_util.to_base64(image_path)
    if err:
        return {"error": err}
    return gemini.call(prompt or gemini.DESCRIBE_PROMPT, b64, mime)


def full_analysis(image_path: str) -> dict:
    """Complete analysis: metadata + description + OCR.

    Returns a single text blob with all information.
    """
    parts = []
    path_obj = Path(image_path).expanduser().resolve()
    path = str(path_obj)

    # Metadata
    meta = img_util.get_metadata(path)
    meta_lines = ["📐 IMAGE METADATA", "━" * 60]
    if meta.get("format"):
        meta_lines.append(f"  • Format: {meta['format']}")
    if meta.get("width") and meta.get("height"):
        meta_lines.append(f"  • Dimensions: {meta['width']} × {meta['height']} px")
    if meta.get("mode"):
        meta_lines.append(f"  • Color mode: {meta['mode']}")
    if meta.get("aspect_ratio"):
        meta_lines.append(f"  • Aspect ratio: {meta['aspect_ratio']:.2f}")
    if meta.get("file_size_kb"):
        meta_lines.append(f"  • File size: {meta['file_size_kb']:.1f} KB")
    parts.append("\n".join(m for m in meta_lines if m))

    # Description via Gemini
    b64, err, mime = img_util.to_base64(path)
    if not err:
        desc = gemini.call(gemini.DESCRIBE_PROMPT, b64, mime)
        if "text" in desc:
            parts.append(
                f"\n🖼️  VISUAL DESCRIPTION\n{'━' * 60}\n{desc['text']}"
            )
    else:
        parts.append(f"\n⚠️  {err}")

    # OCR — always extract, add if non-empty
    ocr_result = extract_text(path)
    if "text" in ocr_result and not _is_empty_result(ocr_result["text"]):
        parts.append(
            f"\n📄 TEXT CONTENT\n{'━' * 60}\n{ocr_result['text']}"
        )

    parts.append(f"\n{'━' * 60}\n📍 {path_obj}")
    return {"text": "\n".join(parts)}


# ── Internal ────────────────────────────────────────────────────────────────


def _ocr_extract(image_path: str) -> dict | None:
    """Run PaddleOCR on the given image. Returns {"text": str, "confidence": float} or None."""
    try:
        ocr = _get_paddle()
        if ocr is None:
            return None

        t0 = time.time()
        raw = ocr.ocr(image_path, cls=True)
        elapsed = time.time() - t0

        if not raw or raw == [[]] or raw == [[None]]:
            return None

        lines = []
        confidences = []
        for page in raw:
            if not page:
                continue
            for line_info in page:
                if line_info is None:
                    continue
                bbox, (text, conf) = line_info
                if text and conf:
                    lines.append(text.strip())
                    confidences.append(conf)

        if not lines:
            return None

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        text = "\n".join(lines)
        log.info("PaddleOCR: %d lines, %.1f conf, %.2fs", len(lines), avg_conf * 100, elapsed)
        return {"text": text, "confidence": avg_conf}
    except ImportError as e:
        log.error("PaddleOCR import error: %s", e)
        return None
    except Exception as e:
        log.warning("PaddleOCR runtime error: %s", e)
        return None
