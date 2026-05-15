"""
opencode-vision: Model-agnostic image analysis via MCP for OpenCode.

An MCP (Model Context Protocol) server that gives vision capabilities to
text-only AI models. Uses PaddleOCR (deep learning, SOTA accuracy) as the
primary OCR engine with Google Gemini Vision API as fallback for handwritten
or degraded text — returning plain text that any model can consume.

Architecture:
    Text-only model → calls MCP tool → opencode-vision-server →
    PaddleOCR (primary) + Gemini (fallback) → text description

Modules:
    server.py    MCP server — JSON-RPC stdio transport, tool routing
    mcp.py       MCP protocol framing and message helpers
    ocr.py       OCR engine — PaddleOCR + Gemini fallback
    gemini.py    Google Gemini Vision API client
    image.py     Image processing — resize, encode, metadata

Quick start:
    pip install opencode-vision
    # Set GOOGLE_API_KEY in ~/.config/opencode/.env
    # Add to opencode.json MCP config → see README

Requirements:
    - Python >= 3.10
    - Google Gemini API key (free at https://aistudio.google.com/)
    - Optional: paddleocr for local OCR (pip install paddleocr)
    - Optional: Pillow for image resizing (pip install pillow)

License: MIT
"""

__version__ = "2.0.0"
__author__ = "Nicolás Ríos Herrera"
__email__ = "nrios@icesi.edu.co"
__license__ = "MIT"
