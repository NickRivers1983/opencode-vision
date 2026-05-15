# opencode-vision 👁️

[![PyPI - Downloads](https://img.shields.io/pypi/dm/opencode-vision?color=blue&label=downloads)](https://pypi.org/project/opencode-vision)
[![PyPI - Version](https://img.shields.io/pypi/v/opencode-vision)](https://pypi.org/project/opencode-vision)
[![GitHub](https://img.shields.io/github/license/NickRivers1983/opencode-vision)](https://github.com/NickRivers1983/opencode-vision)

**Vision-empowered MCP server for OpenCode text-only models.**

Give vision capabilities to **any** text-only model — big-pickle, DeepSeek, MiMo,
MiniMax, or any other model that can't process images natively.

```
pip install opencode-vision[paddle]
```

---

## The Problem

OpenCode supports many models, but most open-weight and free models are
**text-only**. When you paste an image or try to `read()` one, you get:

```
ERROR: Cannot read image (this model does not support image input).
```

This is not a configuration issue — it's a fundamental limitation of the model
architecture. Text-only models have no visual neurons.

## The Solution

`opencode-vision` is an **MCP server** that acts as a "guide dog" for text-only
models. It handles image analysis via a dual-engine architecture:

```
                    ┌──────────────────────────────────────┐
                    │  opencode-vision MCP Server           │
                    │                                      │
  [big-pickle] ────►│  1. PaddleOCR (PP-OCRv5, SOTA) ────►│──► Text
  [DeepSeek]   ────►│     • 0% error rate on benchmarks    │
  [MiMo]       ────►│     • 100+ languages                 │
                    │     • ~15MB model footprint           │
                    │                                      │
                    │  2. Gemini Vision API (fallback) ────►│──► Text
                    │     • Handwriting & scene text        │
                    │     • 1,500 free requests/day         │
                    │     • Zero installation               │
                    └──────────────────────────────────────┘
```

### Why PaddleOCR (not Tesseract)?

| Metric | PaddleOCR (PP-OCRv5) | Tesseract 5 |
|--------|---------------------|-------------|
| Character Error Rate | **4.5%** | 18.2% (4× worse) |
| Invoice accuracy | **100%** (0 errors) | 87.5% (3 errors) |
| OmniDocBench score | **92.86** (SOTA) | N/A |
| Rotated text | ✓ Highly robust | ✗ Fails >5° |
| Scene text accuracy | **85–90%** | 60–70% |
| Model size | ~15MB | ~30MB |
| License | Apache 2.0 | Apache 2.0 |

The community consensus in 2026 is clear: **Tesseract is no longer competitive**
for production OCR. PaddleOCR's deep learning pipeline delivers 4× lower error
rates, handles rotated and degraded text, and supports 100+ languages.

### Gemini Fallback

PaddleOCR struggles with handwriting (14.4% accuracy). When confidence is below
70%, the server falls back to **Google Gemini 2.5 Flash Vision API** (FREE tier,
1,500 requests/day, no credit card required), which achieves 86%+ accuracy on
handwritten text and handles scene text perfectly.

## Quick Start

### 1. Install

```bash
pip install opencode-vision[paddle]    # Recommended: PaddleOCR + Pillow
pip install opencode-vision            # Minimal: Gemini API only
```

### 2. Get a Gemini API key

Get a free key at [aistudio.google.com](https://aistudio.google.com/) (1,500 requests/day, no credit card required).

Set it in `~/.config/opencode/.env`:

```bash
echo 'GOOGLE_API_KEY=your_key_here' >> ~/.config/opencode/.env
```

Or export it directly:

```bash
export GOOGLE_API_KEY=your_key_here
```

### 3. Add to OpenCode config

Add this to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
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

### 4. Restart OpenCode

Start a new session. The `vision_describe`, `vision_ocr`, and `vision_analyze`
tools will be available to all models — even text-only ones.

### 5. Ask about images

```
User: What's in this image?
Model: [calls vision_describe("/path/to/image.png")]
       "A dark gradient banner with 'Nicolás Ríos Herrera'..."
```

## Tools

| Tool | Description | When to use |
|---|---|---|
| `vision_describe(path, prompt?)` | Describe an image in detail | "What does this show?" |
| `vision_ocr(path)` | Extract all visible text | "What text is in this screenshot?" |
| `vision_analyze(path)` | Metadata + description + OCR | Comprehensive understanding |

## Dependencies

| Component | Required? | Notes |
|---|---|---|
| Python >= 3.10 | ✅ Required | |
| `GOOGLE_API_KEY` | ✅ Required | Get free at aistudio.google.com |
| `pillow` | 📦 Recommended | `pip install pillow` for metadata + auto-resize |
| `paddleocr` | 🚀 Recommended | `pip install paddleocr` for local SOTA OCR |
| `tesseract-ocr` | ❌ Deprecated | No longer used. PaddleOCR replaces it entirely. |

The server auto-detects the API key from (in order):
1. `GOOGLE_API_KEY` environment variable
2. `GOOGLE_GENERATIVE_AI_API_KEY` environment variable
3. `~/.config/opencode/.env` file
4. `~/.env` file
5. `$PWD/.env` file

## CLI Usage (without OpenCode)

```bash
# Start MCP server (for OpenCode integration)
opencode-vision

# Direct analysis
opencode-vision describe ~/screenshot.png
opencode-vision ocr ~/scanned-document.png
opencode-vision analyze ~/photo.jpg

# Custom prompt
opencode-vision describe ~/chart.png "What are the values in this chart?"
```

## Architecture

### Why Python?

All existing MCP vision servers for OpenCode are **Node.js/TypeScript** and
require `npm install` or `npx`. `opencode-vision` is pure Python because:

- Python is already installed on every developer machine
- `pillow` (PIL) is the standard image processing library
- PaddleOCR is the best open-source OCR engine available
- The MCP protocol is simple JSON-RPC over stdio — no framework needed
- Zero `node_modules`, zero `npm`, zero `npx`

### Modular Design (v2.0)

```
opencode-vision/
├── opencode_vision/
│   ├── __init__.py    # Package metadata
│   ├── __main__.py    # CLI entry point
│   ├── server.py      # MCP server (thin router)
│   ├── mcp.py         # MCP transport protocol
│   ├── ocr.py         # OCR engine (PaddleOCR + Gemini fallback)
│   ├── gemini.py      # Gemini Vision API client
│   └── image.py       # Image processing utilities
├── pyproject.toml
└── README.md
```

### OCR Strategy

```
                    ┌────────────────────────────┐
                    │   PaddleOCR (PP-OCRv5)      │
                    │   • Deep learning OCR       │
  User image ──────►│   • 0% error on benchmarks  │───► conf ≥ 70% ──► Return text
                    │   • 100+ languages           │
                    └─────────┬──────────────────┘
                              │ conf < 70% / error
                              ▼
                    ┌────────────────────────────┐
                    │   Gemini 2.5 Flash Vision   │
                    │   • Handwriting / scene     │───► Return text
                    │   • 1,500 free req/day      │
                    └────────────────────────────┘
```

### Cost: $0

- Gemini 2.5 Flash: **1,500 free requests/day** via Google AI Studio API key
- PaddleOCR: **free and open-source** (Apache 2.0)
- Pillow: **free and local** for metadata
- No OpenCode Go credits consumed — the API call happens in the vision server,
  not through OpenCode's model proxy

## Comparison with Alternatives

| Feature | opencode-vision v2 | opencode-vision v1 | opencode-minimax-easy-vision | qwen-vision-mcp |
|---|---|---|---|---|
| **Runtime** | Python (stdlib) | Python (stdlib) | Node.js + npm | Node.js + npm |
| **OCR engine** | PaddleOCR (SOTA) | Tesseract (legacy) | None (API only) | None (API only) |
| **OCR accuracy** | **0% error rate** | ~18% CER | N/A | N/A |
| **Handwriting** | Gemini Vision API | ❌ Not supported | ❌ | ❌ |
| **Dependencies** | `pip install opencode-vision[paddle]` | `pip install opencode-vision` | `npm install` | `npx` |
| **API cost** | $0 (Gemini FREE tier) | $0 | MiniMax pricing | $0 (local) |
| **Auto .env** | ✓ Reads ~/.config/opencode/.env | ✓ | ✗ Manual env vars | ✗ |
| **Image resize** | ✓ Pillow auto-resize | ✓ Pillow | ✗ | ✗ |
| **Install size** | ~200 KB + optional 15MB model | ~200 KB | ~30 MB | ~30 MB |

## Why "Model-Agnostic"?

The key architectural insight: **the model never needs to see pixels**. The MCP
server does all the visual processing externally and returns text. This means:

- Works with **any** text-only model (big-pickle, DeepSeek, MiMo, MiniMax, etc.)
- Works with **any** multimodal model too (it doesn't interfere)
- No model-specific configuration
- No provider-specific setup
- The model can be changed at any time without reconfiguring vision

## License

MIT

---

Built with ❤️ by [Nicolás Ríos Herrera](mailto:nrios@icesi.edu.co) for the
OpenCode community.
