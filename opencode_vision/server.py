#!/usr/bin/env python3
"""
opencode-vision-server: MCP (Model Context Protocol) server for model-agnostic image analysis.

PROBLEM:
  OpenCode models like big-pickle and DeepSeek V4 Pro are text-only — they
  cannot process image inputs even when the `read` tool returns them.

SOLUTION:
  This MCP server acts as a "guide dog" for text-only models. It runs as an
  independent process with access to:
    • Google Gemini Vision API (FREE tier via GOOGLE_API_KEY)
    • Local tesseract OCR (fast, private, offline)
    • Pillow for image metadata and resizing

  The server exposes tools that return PLAIN TEXT — any model, text-only or
  multimodal, can consume the output. The model never needs to "see" pixels.

ARCHITECTURE:
  ┌─ Model (text-only) ─┐     MCP stdio JSON-RPC     ┌─ Vision Server ──────┐
  │  vision_describe()  │ ◄────────────────────────► │  Google Gemini API   │
  │  vision_ocr()       │     tool call / result      │  + tesseract OCR     │
  │  vision_analyze()   │                             └──────────────────────┘
  └─────────────────────┘

DEPENDENCIES (all pre-installed or stdlib):
  • Python >= 3.10
  • urllib (stdlib)
  • Pillow >= 10 (optional, for metadata + resize)
  • tesseract-ocr (optional, for local OCR)

USAGE (OpenCode MCP config):
  Add to your opencode.json:
  ```json
  {
    "mcp": {
      "vision": {
        "type": "local",
        "command": ["python3", "-m", "opencode_vision.server"],
        "enabled": true,
        "timeout": 30000
      }
    }
  }
  ```
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR = Path.home() / ".local" / "share" / "opencode-vision"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "vision-server.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger(__name__)

# ── Optional imports ────────────────────────────────────────────────────────
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    log.warning("Pillow not installed. Install: pip install pillow")


# ── MCP Protocol (JSON-RPC 2.0 over stdio) ─────────────────────────────────--

def mcp_send(msg: dict) -> None:
    """Send a JSON-RPC message with MCP Content-Length framing."""
    try:
        content = json.dumps(msg, ensure_ascii=False, default=str)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        sys.stdout.buffer.write(header.encode("utf-8") + content.encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception as e:
        log.error("mcp_send failed: %s", e)


def mcp_recv() -> dict | None:
    """Receive a JSON-RPC message from stdin with MCP framing."""
    try:
        content_length = 0
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")
            if decoded.lower().startswith("content-length:"):
                content_length = int(decoded.split(":", 1)[1].strip())

        if content_length > 0:
            body = sys.stdin.buffer.read(content_length)
            return json.loads(body.decode("utf-8"))
        return None
    except json.JSONDecodeError:
        return None
    except Exception as e:
        log.error("mcp_recv error: %s", e)
        return None


def mcp_result(id_val, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id_val, "result": result}


def mcp_error(id_val, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_val, "error": {"code": code, "message": message}}


# ── Environment / API Key ───────────────────────────────────────────────────

def get_api_key() -> str | None:
    """Get Google API key: env vars → ~/.config/opencode/.env → ~/.env → ./.env"""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
    if key:
        return key

    env_paths = [
        Path.home() / ".config" / "opencode" / ".env",
        Path.home() / ".env",
        Path.cwd() / ".env",
    ]

    for env_file in env_paths:
        if env_file.exists():
            try:
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        if k in ("GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
                            log.info("Found API key in %s", env_file)
                            return v
            except Exception as e:
                log.warning("Error reading %s: %s", env_file, e)
    return None


# ── Image Processing ────────────────────────────────────────────────────────

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


def image_to_base64(image_path: str) -> tuple:
    """Read image, optionally resize → (b64, error, mime)."""
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
            import io

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


# ── Gemini API Call ─────────────────────────────────────────────────────────

DESCRIBE_DEFAULT_PROMPT = """Describe this image comprehensively and objectively. Include:
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


def _call_gemini(b64_data: str, mime_type: str, prompt: str) -> dict:
    """Call Google Gemini 2.5 Flash Vision API. Returns {"text":"..."} or {"error":"..."}."""
    api_key = get_api_key()
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found. Set in ~/.config/opencode/.env or export it."}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": b64_data}},
            ]
        }],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
        "safetySettings": [
            {"category": c, "threshold": "BLOCK_NONE"}
            for c in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ],
    }

    import urllib.request
    import urllib.error

    for attempt in range(3):
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
                return {"error": f"[EMPTY: finishReason={candidates[0].get('finishReason','?')}]"}
            return {"text": text.strip()}

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"error": f"HTTP {e.code}: {body[:300]}"}
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"error": f"{type(e).__name__}: {e}"}

    return {"error": "Failed after 3 retries"}


# ── Tool Implementations ────────────────────────────────────────────────────

def tool_describe(image_path: str, prompt: str | None = None) -> dict:
    """Describe an image using Gemini Vision."""
    b64, err, mime = image_to_base64(image_path)
    if err:
        return {"error": err}
    return _call_gemini(b64, mime, prompt or DESCRIBE_DEFAULT_PROMPT)


def tool_ocr(image_path: str) -> dict:
    """Extract text: local tesseract → Gemini fallback."""
    # 1. Try local tesseract
    try:
        proc = subprocess.run(
            ["tesseract", str(Path(image_path).expanduser().resolve()),
             "stdout", "-l", "spa+eng", "--psm", "3"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            text = proc.stdout.strip()
            if len(text) > 5:
                return {"text": text}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        log.warning("tesseract failed: %s", e)

    # 2. Gemini fallback
    b64, err, mime = image_to_base64(image_path)
    if err:
        return {"error": err}
    return _call_gemini(b64, mime, OCR_PROMPT)


def tool_analyze(image_path: str) -> dict:
    """Full analysis: metadata + description + OCR."""
    parts = []
    path = Path(image_path).expanduser().resolve()

    # Metadata
    meta_lines = ["📐 IMAGE METADATA", "━" * 60]
    if HAS_PIL:
        try:
            img = Image.open(path)
            meta_lines.extend([
                f"  • Format: {img.format or 'unknown'}",
                f"  • Dimensions: {img.width} × {img.height} px",
                f"  • Color mode: {img.mode}",
                f"  • Aspect ratio: {img.width / img.height:.2f}" if img.height else "",
            ])
        except Exception:
            pass
    try:
        kb = path.stat().st_size / 1024
        meta_lines.append(f"  • File size: {kb:.1f} KB")
    except Exception:
        pass
    parts.append("\n".join(m for m in meta_lines if m))

    # Description
    b64, err, mime = image_to_base64(image_path)
    if not err:
        desc = _call_gemini(b64, mime, DESCRIBE_DEFAULT_PROMPT)
        if "text" in desc:
            parts.append(f"\n🖼️  VISUAL DESCRIPTION\n{'━' * 60}\n{desc['text']}")
    else:
        parts.append(f"\n⚠️  {err}")

    # OCR
    ocr = tool_ocr(image_path)
    if "text" in ocr and ocr["text"] not in ("[No text detected]", "[No text detected in image]"):
        parts.append(f"\n📄 TEXT CONTENT\n{'━' * 60}\n{ocr['text']}")

    parts.append(f"\n{'━' * 60}\n📍 {path}")
    return {"text": "\n".join(parts)}


# ── Tool Registry ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "vision_describe",
        "description": "Describe an image in detail. Returns composition, colors, text, objects, and context. Pass an optional `prompt` for specific questions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to image file"},
                "prompt": {"type": "string", "description": "Optional custom question about the image"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "vision_ocr",
        "description": "Extract text from an image using OCR. Uses local tesseract first (fast, private), falls back to Gemini Vision API for difficult cases. Supports Spanish + English.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to image file containing text"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "vision_analyze",
        "description": "Complete image analysis: file metadata + visual description + OCR text extraction. Best for comprehensive understanding of any image.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to image file"},
            },
            "required": ["image_path"],
        },
    },
]


# ── MCP Handlers ────────────────────────────────────────────────────────────

def handle_initialize(msg):
    return mcp_result(msg["id"], {
        "protocolVersion": "0.1.0",
        "serverInfo": {"name": "opencode-vision-server", "version": "1.0.0"},
        "capabilities": {"tools": {}},
    })


def handle_list_tools(msg):
    return mcp_result(msg["id"], {"tools": TOOLS})


def handle_call_tool(msg):
    params = msg.get("params", {})
    name = params.get("name", "")
    args = params.get("arguments", {})

    path = args.get("image_path", "")
    if not path:
        return mcp_error(msg["id"], -32000, "Missing required: image_path")

    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        return mcp_error(msg["id"], -32000,
                         f"File not found: {path_obj}\nUse an absolute path to an existing image.")

    try:
        if name == "vision_describe":
            result = tool_describe(str(path_obj), args.get("prompt"))
        elif name == "vision_ocr":
            result = tool_ocr(str(path_obj))
        elif name == "vision_analyze":
            result = tool_analyze(str(path_obj))
        else:
            return mcp_error(msg["id"], -32601, f"Unknown tool: {name}")

        if "error" in result:
            return mcp_error(msg["id"], -32000, result["error"])

        return mcp_result(msg["id"], {
            "content": [{"type": "text", "text": result.get("text", str(result))}],
        })
    except Exception as e:
        log.error("Tool crash: %s\n%s", e, traceback.format_exc())
        return mcp_error(msg["id"], -32000, f"{type(e).__name__}: {e}")


# ── Main Loop ───────────────────────────────────────────────────────────────

METHOD_HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_list_tools,
    "tools/call": handle_call_tool,
}


def main() -> None:
    """Start the MCP server. Reads JSON-RPC from stdin, writes to stdout."""
    # Banner
    api_status = "✓" if get_api_key() else "✗ (set GOOGLE_API_KEY)"
    log.info("─" * 50)
    log.info("opencode-vision-server v1.0")
    log.info("Python: %s | Pillow: %s | tesseract: %s",
             sys.version.split()[0],
             "✓" if HAS_PIL else "✗",
             "✓" if subprocess.run(["which", "tesseract"], capture_output=True).returncode == 0 else "✗")
    log.info("Gemini API key: %s", api_status)
    log.info("Log: %s", LOG_FILE)
    log.info("─" * 50)

    while True:
        try:
            msg = mcp_recv()
            if msg is None:
                break

            method = msg.get("method", "")
            mid = msg.get("id")

            if method == "notifications/initialized":
                pass  # no response needed
            elif method in ("shutdown", "exit"):
                break
            elif method in METHOD_HANDLERS:
                mcp_send(METHOD_HANDLERS[method](msg))
            elif mid is not None:
                mcp_send(mcp_error(mid, -32601, f"Unknown: {method}"))
        except Exception as e:
            log.error("Loop error: %s", e)

    log.info("Shutdown complete")


if __name__ == "__main__":
    main()
