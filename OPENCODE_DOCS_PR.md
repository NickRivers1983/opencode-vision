# PR Content for OpenCode Docs — MCP Servers Page

This is the content to add to the OpenCode docs MCP servers page.
The file to edit is: `packages/web/src/content/docs/mcp-servers.mdx`
at `github.com/anomalyco/opencode`

## Content to add (after the Grep by Vercel example)

Add a new section:

---

### opencode-vision

Add vision capabilities to **any** text-only model. When your model can't process images natively (e.g., big-pickle, DeepSeek, MiMo), this MCP server handles image analysis via **Google Gemini Vision API** (FREE tier) and **local tesseract OCR**, returning text descriptions that any model can understand.

**Requirements:**
- Python >= 3.10
- Google Gemini API key (get one free at [aistudio.google.com](https://aistudio.google.com/))

```json title="opencode.json"
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

**Setup:**

1. Install the server:
   ```bash
   pip install opencode-vision
   ```
2. Set your Gemini API key in `~/.config/opencode/.env`:
   ```bash
   echo 'GOOGLE_API_KEY=your_key_here' >> ~/.config/opencode/.env
   ```
3. Restart OpenCode.

**Available tools:**

| Tool | Description |
|------|-------------|
| `vision_describe(path, prompt?)` | Describe an image's composition, colors, text, and context |
| `vision_ocr(path)` | Extract text from images via tesseract + Gemini fallback |
| `vision_analyze(path)` | Full analysis: metadata + description + OCR |

Learn more at [github.com/nriosdev/opencode-vision](https://github.com/nriosdev/opencode-vision) or `pip install opencode-vision`.
