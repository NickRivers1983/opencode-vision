#!/usr/bin/env python3
"""
opencode-vision-server: MCP server for model-agnostic image analysis.

Gives vision capabilities to text-only models (big-pickle, DeepSeek, etc.)
by providing MCP tools that return PLAIN TEXT descriptions.

Architecture:
    ┌─ Model (text-only) ─┐     MCP stdio JSON-RPC     ┌─ Vision Server ──────┐
    │  vision_describe()  │ ◄────────────────────────► │  Image analysis via  │
    │  vision_ocr()       │     tool call / result      │  PaddleOCR + Gemini  │
    │  vision_analyze()   │                             └──────────────────────┘
    └─────────────────────┘

Usage (OpenCode MCP config):
    Add to opencode.json:
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

import logging
import sys
import traceback
from pathlib import Path

from opencode_vision import gemini, image as img_util, mcp, ocr

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

# ── Tool Registry ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "vision_describe",
        "description": (
            "Describe an image in detail. Returns composition, colors, "
            "visible text, objects, and context. Pass an optional `prompt` "
            "for specific questions about the image."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to image file",
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional custom question about the image",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "vision_ocr",
        "description": (
            "Extract text from an image using OCR. Uses PaddleOCR "
            "(deep learning, 0% error rate on benchmarks) as primary engine, "
            "with Gemini Vision API fallback for handwritten or degraded text."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to image file containing text",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "vision_analyze",
        "description": (
            "Complete image analysis: file metadata + visual description + "
            "OCR text extraction. Best for comprehensive understanding of "
            "any image."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to image file",
                },
            },
            "required": ["image_path"],
        },
    },
]

# ── MCP Handlers ────────────────────────────────────────────────────────────


def _resolve_path(image_path: str) -> tuple[Path, str | None]:
    """Resolve and validate an image path. Returns (path, error)."""
    path_obj = Path(image_path).expanduser().resolve()
    if not path_obj.exists():
        return path_obj, (
            f"File not found: {path_obj}\n"
            f"Use an absolute path to an existing image."
        )
    if not path_obj.is_file():
        return path_obj, f"Not a file: {path_obj}"
    return path_obj, None


def handle_initialize(msg):
    return mcp.result(msg["id"], {
        "protocolVersion": "0.1.0",
        "serverInfo": {"name": "opencode-vision-server", "version": "2.0.0"},
        "capabilities": {"tools": {}},
    })


def handle_list_tools(msg):
    return mcp.result(msg["id"], {"tools": TOOLS})


def handle_call_tool(msg):
    params = msg.get("params", {})
    name = params.get("name", "")
    args = params.get("arguments", {})

    image_path = args.get("image_path", "")
    if not image_path:
        return mcp.error(msg["id"], -32000, "Missing required: image_path")

    path_obj, err = _resolve_path(image_path)
    if err:
        return mcp.error(msg["id"], -32000, err)

    try:
        if name == "vision_describe":
            result = ocr.describe_image(str(path_obj), args.get("prompt"))
        elif name == "vision_ocr":
            result = ocr.extract_text(str(path_obj))
        elif name == "vision_analyze":
            result = ocr.full_analysis(str(path_obj))
        else:
            return mcp.error(msg["id"], -32601, f"Unknown tool: {name}")

        if "error" in result:
            return mcp.error(msg["id"], -32000, result["error"])

        return mcp.result(msg["id"], {
            "content": [{"type": "text", "text": result.get("text", str(result))}],
        })
    except Exception as e:
        log.error("Tool crash: %s\n%s", e, traceback.format_exc())
        return mcp.error(msg["id"], -32000, f"{type(e).__name__}: {e}")


# ── Main Loop ───────────────────────────────────────────────────────────────

METHOD_HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_list_tools,
    "tools/call": handle_call_tool,
}


def main() -> None:
    """Start the MCP server. Reads JSON-RPC from stdin, writes to stdout."""

    # Banner
    api_key = gemini.get_api_key()
    api_status = "✓" if api_key else "✗ (set GOOGLE_API_KEY)"

    log.info("─" * 50)
    log.info("opencode-vision-server v2.0")
    log.info("Python:      %s", sys.version.split()[0])
    log.info("OCR engine:  %s", "PaddleOCR + Gemini" if ocr.HAS_PADDLE else "Gemini Vision API")
    log.info("Pillow:      %s", "✓" if img_util.HAS_PIL else "✗ (pip install pillow)")
    log.info("Gemini API:  %s", api_status)
    log.info("Log:         %s", LOG_FILE)
    log.info("─" * 50)

    while True:
        try:
            msg = mcp.recv()
            if msg is None:
                break

            method = msg.get("method", "")
            mid = msg.get("id")

            if method == "notifications/initialized":
                pass  # no response needed
            elif method in ("shutdown", "exit"):
                break
            elif method in METHOD_HANDLERS:
                mcp.send(METHOD_HANDLERS[method](msg))
            elif mid is not None:
                mcp.send(mcp.error(mid, -32601, f"Unknown: {method}"))
        except Exception as e:
            log.error("Loop error: %s", e)

    log.info("Shutdown complete")


if __name__ == "__main__":
    main()
