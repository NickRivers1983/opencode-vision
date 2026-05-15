# opencode-vision 👁️

[![PyPI - Downloads](https://img.shields.io/pypi/dm/opencode-vision?color=blue&label=downloads)](https://pypi.org/project/opencode-vision)
[![PyPI - Version](https://img.shields.io/pypi/v/opencode-vision)](https://pypi.org/project/opencode-vision)
[![GitHub](https://img.shields.io/github/license/NickRivers1983/opencode-vision)](https://github.com/NickRivers1983/opencode-vision)

**Model-agnostic image analysis via MCP for OpenCode.**

Give vision capabilities to **any** text-only model — big-pickle, DeepSeek, MiMo,
MiniMax, or any other model that can't process images natively.

```
pip install opencode-vision
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
models. It runs as an independent process and handles image analysis via:

1. **Google Gemini Vision API** (FREE tier, 1,500 requests/day)
2. **Local tesseract OCR** (fast, private, works offline)

It returns **plain text descriptions** that any model can understand — no vision
capabilities needed on the model's side.

```
                    ┌─────────────────────┐
                    │  opencode-vision     │
  [big-pickle] ────►│  MCP Server          │────► Google Gemini API
  [DeepSeek]   ────►│  (Python process)    │────► tesseract OCR (local)
  [MiMo]       ────►│  Returns TEXT only   │
                    └─────────────────────┘
```

## Quick Start

### 1. Install

```bash
pip install opencode-vision
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

## Requirements

| Component | Required? | Notes |
|---|---|---|
| Python >= 3.10 | ✅ Required | |
| `GOOGLE_API_KEY` | ✅ Required | Get free at aistudio.google.com |
| `pillow` | 📦 Recommended | `pip install pillow` for metadata + auto-resize |
| `tesseract-ocr` | 🔧 Recommended | For local OCR. `apt install tesseract-ocr` or `brew install tesseract` |

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
- tesseract has first-class Python bindings
- Zero `node_modules`, zero `npm`, zero `npx`
- The MCP protocol is simple JSON-RPC over stdio — no framework needed

### Hybrid OCR Strategy

```
                    ┌──────────────┐
  User image ──────►│ tesseract    │───► Text found? ──► Return
                    │ (local,      │
                    │  private,    │
                    │  offline)    │
                    └──────┬───────┘
                           │ No text / failed
                           ▼
                    ┌──────────────┐
                    │ Gemini       │───► Return
                    │ Vision API   │
                    │ (FREE tier)  │
                    └──────────────┘
```

### Cost: $0

- Gemini 2.5 Flash: **1,500 free requests/day** via Google AI Studio API key
- tesseract: **free and local** for OCR
- Pillow: **free and local** for metadata
- No OpenCode Go credits consumed — the API call happens in the vision server,
  not through OpenCode's model proxy

## Comparison with Alternatives

| Feature | opencode-vision | opencode-minimax-easy-vision | qwen-vision-mcp | opencode-image-proxy |
|---|---|---|---|---|
| **Runtime** | Python (stdlib) | Node.js + npm | Node.js + npm | Node.js |
| **Dependencies** | `pip install opencode-vision` | `npm install` + MiniMax API key | `npx` + Ollama (6GB RAM) | `npx` |
| **OCR** | Local tesseract + Gemini fallback | None (API only) | None (API only) | None (API only) |
| **API cost** | $0 (Gemini FREE tier) | MiniMax pricing | $0 (local) or Ollama Cloud | OpenCode credits |
| **Auto .env** | ✓ Reads ~/.config/opencode/.env | ✗ Manual env vars | ✗ Manual env vars | ✗ Manual config |
| **Image resize** | ✓ Pillow auto-resize | ✗ | ✗ | ✗ |
| **Install size** | ~200 KB (pure Python) | ~30 MB (node_modules) | ~30 MB + ~6GB (Ollama model) | ~30 MB |

## Why "Model-Agnostic"?

The key architectural insight: **the model never needs to see pixels**. The MCP
server does all the visual processing externally and returns text. This means:

- Works with **any** text-only model (big-pickle, DeepSeek, MiMo, MiniMax, GLM, etc.)
- Works with **any** multimodal model too (it doesn't interfere)
- No model-specific configuration
- No provider-specific setup
- The model can be changed at any time without reconfigureing vision

## License

MIT

---

Built with ❤️ by [Nicolás Ríos Herrera](mailto:nrios@icesi.edu.co) for the
OpenCode community.
