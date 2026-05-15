#!/usr/bin/env python3
"""
Entry point for `python -m opencode_vision` and `opencode-vision` CLI.

When called without arguments, starts the MCP server (stdio transport).
When called with `describe|ocr|analyze <image_path>`, runs the tool directly
using the same OCR engine pipeline (PaddleOCR + Gemini fallback).

Examples:
    # Start MCP server (for OpenCode integration)
    opencode-vision

    # CLI mode — describe an image
    opencode-vision describe ~/screenshot.png
    opencode-vision analyze ~/photo.jpg
    opencode-vision ocr ~/document.png

    # CLI with custom prompt
    opencode-vision describe ~/chart.png "What values does this chart show?"
"""

import json
import sys
from pathlib import Path

# Add parent to path for direct execution
_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent.parent))

from opencode_vision.server import main as mcp_main
from opencode_vision import ocr


def main():
    """CLI entry point: MCP server by default, direct tool if args given."""
    if len(sys.argv) >= 3 and sys.argv[1] in ("describe", "ocr", "analyze"):
        command = sys.argv[1]
        image_path = sys.argv[2]

        if command == "describe":
            prompt = sys.argv[3] if len(sys.argv) > 3 else None
            result = ocr.describe_image(image_path, prompt)
        elif command == "ocr":
            result = ocr.extract_text(image_path)
        elif command == "analyze":
            result = ocr.full_analysis(image_path)
        else:
            result = {"error": f"Unknown command: {command}"}

        if "error" in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)
        print(result.get("text", json.dumps(result, indent=2)))
    else:
        # Start MCP server
        mcp_main()


if __name__ == "__main__":
    main()
