# Drafts de Publicity — opencode-vision

Copia y pega donde quieras.

---

## Discord (canal #mcp-servers o #showcase)

**Título del post:** opencode-vision — MCP server Python para que modelos text-only vean imágenes

**Cuerpo:**

Hice un MCP server en Python para solucionar un problema: cuando tu modelo de OpenCode no soporta imágenes (big-pickle, DeepSeek, etc.), necesitas un "puente" que analice las imágenes y devuelva texto.

**opencode-vision** usa Google Gemini Vision API (FREE tier, 1500 req/día) + tesseract OCR local para describir imágenes, extraer texto, y hacer análisis completo.

```bash
pip install opencode-vision
```

Y en opencode.json:
```json
"vision": {
  "type": "local",
  "command": ["python3", "-m", "opencode_vision.server"],
  "enabled": true,
  "timeout": 30000
}
```

Tres tools:
- `vision_describe(path)` — describe composición, colores, contexto
- `vision_ocr(path)` — extrae texto (tesseract + fallback Gemini)
- `vision_analyze(path)` — metadata + descripción + OCR

Zero dependencias nuevas si ya tienes Pillow y tesseract. Sin npm, sin node_modules, sin 6GB de modelos locales.

GitHub: https://github.com/NickRivers1983/opencode-vision
PyPI: https://pypi.org/project/opencode-vision

---

## Reddit — r/opencode

**Título:** I built a Python MCP server so text-only LLMs can see images (Gemini + OCR)

**Cuerpo:**

If you use OpenCode with models like big-pickle or DeepSeek that don't support images, here's a solution I built.

**opencode-vision** is a model-agnostic MCP vision server. Instead of trying to make the model multimodal, it acts as a proxy: receives image paths → analyzes via Google Gemini API + local tesseract OCR → returns text descriptions any model can understand.

**Why I built it:**
- All existing MCP vision servers are Node.js/TS (npm install, npx, node_modules)
- Wanted something that works with just `pip install`
- Google Gemini FREE tier = 1500 requests/day, no GPU needed
- Hybrid OCR (tesseract first → Gemini fallback) works offline for simple text extraction

**Quick start:**
```bash
pip install opencode-vision
```

Add to opencode.json:
```json
"vision": {
  "type": "local",
  "command": ["python3", "-m", "opencode_vision.server"],
  "enabled": true,
  "timeout": 30000
}
```

Three tools: `vision_describe`, `vision_ocr`, `vision_analyze`.

Try it with any image and let me know what you think!

https://github.com/NickRivers1983/opencode-vision
https://pypi.org/project/opencode-vision

---

## X / Twitter

**Post 1 (lanzamiento):**

I built an MCP vision server for @opencode_ai so text-only models can finally see images.
 
No Node.js. No npm. Just Python + Gemini FREE tier + local OCR.
 
`pip install opencode-vision`
 
https://github.com/NickRivers1983/opencode-vision

[imagen: screenshot de opencode-vision describiendo una imagen]

**Post 2 (técnico, días después):**

Fun fact: all existing MCP vision servers for OpenCode require Node.js.
 
So I wrote one in Python. Zero new dependencies if you already have Pillow.
 
Hybrid OCR (tesseract → Gemini fallback) means it works offline for text extraction.
 
No GPU needed. No 6GB model downloads.
 
https://github.com/NickRivers1983/opencode-vision

---

## Cuándo publicar cada cosa

| Canal | Timing | Notas |
|-------|--------|-------|
| Discord #mcp-servers | Ahora | Público más técnico, buscan servers |
| Reddit r/opencode | Espera PR merge (días/semana) | La gente revisa, mejor tener el PR aprobado |
| X/Twitter | Cualquier momento | Puedes taggear @opencode_ai |
| GitHub Show and tell | Hecho ✅ | https://github.com/NickRivers1983/opencode-vision/discussions/1 |
