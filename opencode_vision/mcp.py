"""
MCP (Model Context Protocol) transport layer — JSON-RPC 2.0 over stdio.

Provides framing, send/receive, and response helpers for the
Model Context Protocol used by OpenCode to communicate with tools.

Architecture:
    ┌─ OpenCode ─┐     MCP stdio JSON-RPC     ┌─ Vision Server ─┐
    │  tool call  │ ◄────────────────────────► │  tool handlers  │
    │  result     │     Content-Length framing  └─────────────────┘
    └─────────────┘
"""

from __future__ import annotations

import json
import logging
import sys

log = logging.getLogger(__name__)


def send(msg: dict) -> None:
    """Send a JSON-RPC message with MCP Content-Length framing."""
    try:
        content = json.dumps(msg, ensure_ascii=False, default=str)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        sys.stdout.buffer.write(header.encode("utf-8") + content.encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception as e:
        log.error("mcp.send failed: %s", e)


def recv() -> dict | None:
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
        log.error("mcp.recv error: %s", e)
        return None


def result(id_val, result_data: dict) -> dict:
    """Build a success response."""
    return {"jsonrpc": "2.0", "id": id_val, "result": result_data}


def error(id_val, code: int, message: str) -> dict:
    """Build an error response."""
    return {"jsonrpc": "2.0", "id": id_val, "error": {"code": code, "message": message}}
