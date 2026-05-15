"""
Google Gemini Vision API client.

Handles API key detection, image encoding, and Gemini Vision API calls
for image description and OCR fallback.

All calls go through Gemini 2.5 Flash — FREE tier (1,500 requests/day),
no credit card required.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# ── API Key resolution ─────────────────────────────────────────────────────

API_KEY_ENV_VARS = ("GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY")

API_KEY_FILE_PATHS = (
    Path.home() / ".config" / "opencode" / ".env",
    Path.home() / ".env",
    Path.cwd() / ".env",
)


def get_api_key() -> str | None:
    """Resolve Google API key: env vars → .env files.

    Checks in order:
        1. GOOGLE_API_KEY environment variable
        2. GOOGLE_GENERATIVE_AI_API_KEY environment variable
        3. ~/.config/opencode/.env
        4. ~/.env
        5. $PWD/.env
    """
    for var in API_KEY_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val

    for env_file in API_KEY_FILE_PATHS:
        if not env_file.exists():
            continue
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if k in API_KEY_ENV_VARS:
                    log.info("Found API key in %s", env_file)
                    return v
        except Exception as e:
            log.warning("Error reading %s: %s", env_file, e)

    return None


# ── Prompts ─────────────────────────────────────────────────────────────────

DESCRIBE_PROMPT = """Describe this image comprehensively and objectively. Include:
1. Main subject and composition
2. Colors, lighting, visual style
3. ALL visible text (transcribed exactly, preserve original language)
4. People, objects, environment
5. Context and purpose (UI screenshot, photo, diagram, document, etc.)
6. Technical quality and notable visual elements

Be precise. Do not speculate beyond what is visible."""

OCR_PROMPT = """Extract ALL text from this image EXACTLY as it appears.
Preserve the original language, capitalization, line breaks, and formatting.
Return ONLY the extracted text with no commentary.
If there is no readable text, say "[No text detected]"."""

# ── API call ────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

DEFAULT_CONFIG = {
    "temperature": 0.2,
    "maxOutputTokens": 8192,
}

SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds, doubles each retry


def call(prompt: str, b64_data: str, mime_type: str = "image/png") -> dict:
    """Call Google Gemini 2.5 Flash Vision API.

    Args:
        prompt: Text prompt to send alongside the image.
        b64_data: Base64-encoded image data.
        mime_type: MIME type of the image (e.g. "image/png").

    Returns:
        {"text": "..."} on success, or {"error": "..."} on failure.
    """
    api_key = get_api_key()
    if not api_key:
        return {
            "error": (
                "GOOGLE_API_KEY not found. "
                "Set in ~/.config/opencode/.env or export it."
            )
        }

    url = f"{GEMINI_URL}?key={api_key}"

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": b64_data}},
            ]
        }],
        "generationConfig": DEFAULT_CONFIG,
        "safetySettings": [
            {"category": c, "threshold": "BLOCK_NONE"}
            for c in SAFETY_CATEGORIES
        ],
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            candidates = result.get("candidates", [])
            if not candidates:
                reason = result.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
                return {"error": f"[BLOCKED: {reason}]"}

            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if not text:
                finish = candidates[0].get("finishReason", "?")
                return {"error": f"[EMPTY: finishReason={finish}]"}

            return {"text": text.strip()}

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY ** (attempt + 1)
                log.warning("Rate limited, retrying in %ds...", delay)
                time.sleep(delay)
                last_error = {"error": f"HTTP {e.code}: {body[:300]}"}
                continue
            return {"error": f"HTTP {e.code}: {body[:300]}"}

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY ** (attempt + 1)
                log.warning("API error, retrying in %ds: %s", delay, e)
                time.sleep(delay)
                last_error = {"error": f"{type(e).__name__}: {e}"}
                continue
            return {"error": f"{type(e).__name__}: {e}"}

    return last_error or {"error": "Failed after all retries"}
