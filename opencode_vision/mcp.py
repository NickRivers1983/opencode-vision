"""
MCP (Model Context Protocol) transport layer — JSON-RPC 2.0 over stdio.

Provides framing, send/receive, and response helpers for the
Model Context Protocol used by OpenCode to communicate with tools.

Architecture:
    ┌─ OpenCode ─┐     MCP stdio JSON-RPC     ┌─ Vision Server ─┐
    │  tool call  │ ◄────────────────────────► │  tool handlers  │
    │  result     │     newline-delimited JSON  └─────────────────┘
    └─────────────┘
"""

from __future__ import annotations

import json
import logging
import sys

log = logging.getLogger(__name__)


def send(msg: dict) -> None:
    """Send a JSON-RPC message as newline-delimited JSON (MCP stdio transport).

    The MCP stdio transport delimits messages by newlines: one JSON object per
    line, with no framing headers. (Content-Length framing is LSP, a different
    protocol; clients such as OpenCode read line-by-line and never register a
    Content-Length-framed reply, so the request eventually times out with
    ``-32001``.) ``json.dumps`` with default settings never emits embedded
    newlines, so each message stays on a single line as the spec requires.
    """
    try:
        body = json.dumps(msg, ensure_ascii=False, default=str)
        sys.stdout.buffer.write((body + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception as e:
        log.error("mcp.send failed: %s", e)


def recv() -> dict | None:
    """Receive a JSON-RPC message from stdin.

    Supports two transports:
    1. Newline-delimited JSON (MCP stdio standard): one JSON object per line.
    2. Content-Length framing (LSP-style): ``Content-Length: N\\r\\n\\r\\n<body>``,
       accepted for backward compatibility.
    """
    try:
        line = sys.stdin.buffer.readline()
        if not line:
            return None

        decoded = line.decode("utf-8", errors="replace").strip()

        # Transport 2: Content-Length framing (LSP-style), accepted for compat.
        if decoded.lower().startswith("content-length:"):
            content_length = int(decoded.split(":", 1)[1].strip())
            # Skip any remaining headers until the blank separator line.
            while True:
                hdr = sys.stdin.buffer.readline()
                if not hdr or hdr.strip() == b"":
                    break
            body = sys.stdin.buffer.read(content_length)
            return json.loads(body.decode("utf-8"))

        # Transport 1: Newline-delimited JSON (MCP stdio standard).
        if decoded:
            return json.loads(decoded)

        return None
    except json.JSONDecodeError:
        return None
    except Exception as e:
        log.error("mcp.recv error: %s", e)
        return None


def result(id_val, result_data: dict) -> dict:
    """Build a success response."""
    return {"jsonrpc": "2.0", "id": id_val, "result": result_data}


def error(id_val, code: int, message: str) -> dict:
    """Build an error response."""
    return {"jsonrpc": "2.0", "id": id_val, "error": {"code": code, "message": message}}
