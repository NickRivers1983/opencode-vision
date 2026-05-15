#!/usr/bin/env python3
"""
Entry point for `python -m opencode_vision` and `opencode-vision` CLI.

When called without arguments, starts the MCP server (stdio transport).
When called with `describe|ocr|analyze <image_path>`, runs the tool directly.

Examples:
    # Start MCP server (for OpenCode integration)
    opencode-vision

    # CLI mode - describe an image
    opencode-vision describe ~/screenshot.png
    opencode-vision analyze ~/photo.jpg
    opencode-vision ocr ~/document.png
"""

import sys
import json
from pathlib import Path

# Add parent to path for direct execution
_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent.parent))

from opencode_vision.server import main as mcp_main
from opencode_vision.server import tool_describe, tool_ocr, tool_analyze


def main():
    """CLI entry point: MCP server by default, direct tool if args given."""
    if len(sys.argv) >= 3 and sys.argv[1] in ("describe", "ocr", "analyze"):
        command = sys.argv[1]
        image_path = sys.argv[2]
        prompt = sys.argv[3] if len(sys.argv) > 3 and command == "describe" else None

        tool_map = {
            "describe": lambda: tool_describe(image_path, prompt),
            "ocr": lambda: tool_ocr(image_path),
            "analyze": lambda: tool_analyze(image_path),
        }

        result = tool_map[command]()
        if "error" in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)
        print(result.get("text", json.dumps(result, indent=2)))
    else:
        # Start MCP server
        mcp_main()


if __name__ == "__main__":
    main()
