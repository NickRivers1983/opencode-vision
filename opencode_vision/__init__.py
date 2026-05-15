"""
opencode-vision: Model-agnostic image analysis via MCP for OpenCode.

An MCP (Model Context Protocol) server that acts as a "guide dog" for
text-only AI models. When a model like big-pickle or DeepSeek can't process
image inputs, this server handles image analysis via Google Gemini Vision API
and local tesseract OCR, returning text descriptions that any model can
understand.

Architecture:
    Text-only model → calls MCP tool → opencode-vision-server →
    Google Gemini API + tesseract OCR → text description

Usage (OpenCode MCP):
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

Requirements:
    - Python >= 3.10
    - google-genai SDK or direct REST access to Gemini API
    - A Google Gemini API key (get one free at https://aistudio.google.com/)
    - Optional: tesseract-ocr for local text extraction

License: MIT
"""

__version__ = "1.0.0"
__author__ = "Nicolás Ríos Herrera"
__email__ = "nrios@icesi.edu.co"
__license__ = "MIT"
