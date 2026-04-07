<div align="center">

# ObbyAI

**A fully local, self-hosted AI chat platform with multi-GPU routing, RAG, web search, and comprehensive file intelligence.**

[![CI](https://github.com/obnoxiousmods/obbyai/actions/workflows/ci.yml/badge.svg)](https://github.com/obnoxiousmods/obbyai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-7c5cfc.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Vulkan%20%7C%20CUDA-orange)](https://ollama.com)
[![ArangoDB](https://img.shields.io/badge/ArangoDB-3.12-green)](https://arangodb.com)

[Features](#features) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Models](#models) · [Configuration](#configuration) · [Contributing](#contributing)

</div>

---

## What is ObbyAI?

ObbyAI is a **production-quality, self-hosted AI chat interface** built on top of [Ollama](https://ollama.com). It runs entirely on your local hardware with no external API calls — except when you explicitly use the web search tool.

Key design goals:
- **Privacy-first**: all inference runs on your hardware, nothing leaves your network
- **Multi-GPU**: route requests to different machines based on the task (e.g. vision to RTX 2080 Super, fast chat to RX 580)
- **Tool-augmented**: real web search, persistent vector memory, exact math, and file intelligence built in
- **Single-file frontend**: the entire UI is one `index.html` — zero build steps, zero Node.js in production

---

## Features

### 🖥️ Multi-GPU Server Routing
Switch between GPU backends from the header:
- **Local**: AMD RX 580 8GB (Vulkan) — fast 7–8B models
- **Remote**: NVIDIA RTX 2080 Super 8GB (CUDA) — Gemma4 vision, larger models

Each server has its own model list. Requests are proxied through the Python backend — no CORS, no direct browser-to-Ollama.

### 🤖 10+ Curated Models

| Model | Size | Tag | Best For |
|---|---|---|---|
| llama3.1:8b | 4.9 GB | General | Default — reasoning, coding, Q&A |
| qwen2.5:7b | 4.7 GB | General | Multilingual, strong reasoning |
| qwen2.5-coder:7b | 4.7 GB | Coding | Code generation and debugging |
| qwen3:8b | 5.2 GB | Thinking | Hybrid thinking/reasoning mode |
| deepseek-r1:8b | 5.2 GB | Reasoning | Chain-of-thought problems |
| mistral:7b-instruct | 4.4 GB | General | Structured tasks, fast responses |
| mistral-nemo:12b | 7.1 GB | General | More capable, ~7GB VRAM |
| gemma3:4b | 3.3 GB | Fast | Quick responses, efficient |
| phi4-mini:3.8b | 2.5 GB | Fast | Microsoft Phi-4, punches above weight |
| gemma4:latest | 9.6 GB | Vision | Multimodal image+text (RTX 2080S) |

### 🛠️ Tool Calling
Automatic tool use for capable models (llama3.1, qwen2.5, qwen3, mistral, phi4-mini):

| Tool | Trigger | Capability |
|---|---|---|
| 🔍 `web_search` | Current events, lookups, docs | DuckDuckGo, no API key |
| 📚 `rag_search` | "in my documents", context recall | ArangoDB vector search |
| 🧮 `calculator` | Any arithmetic | Safe AST eval, trig, log |
| 🕐 `get_datetime` | Time/date queries | Timezone-aware |

Tool calls are streamed live — you see exactly what the model is searching for and what results it gets.

### 📚 RAG — Retrieval-Augmented Generation
Persistent knowledge base powered by [ArangoDB](https://arangodb.com) + [nomic-embed-text](https://ollama.com/library/nomic-embed-text) embeddings:
- Automatic context injection: relevant chunks from your documents are silently prepended to every query
- Conversation memory: past messages are embedded and retrievable across sessions
- Document library: upload files and they stay in the knowledge base permanently

### 📁 File Intelligence — 40+ Formats

**Images** → Gemma4 vision model performs deep analysis:
- Full OCR — extracts all visible text
- Image description and context
- Table/chart/diagram interpretation
- Code and URL extraction

**Documents:**

| Format | Extraction Method |
|---|---|
| PDF | [pymupdf4llm](https://github.com/pymupdf/RAG) — layout-aware markdown |
| DOCX | python-docx — headings, paragraphs, tables |
| PPTX | python-pptx — per-slide text |
| XLSX/ODS | openpyxl — all sheets as structured text |

**Text & Code** (30+ extensions): direct UTF-8 extraction with encoding detection. Python, JavaScript, TypeScript, Rust, Go, Java, C/C++, SQL, YAML, TOML, JSON, Markdown, and more.

All extracted content is chunked, embedded, and stored in ArangoDB for future retrieval.

### 🎭 Persona Library
7 curated system prompts selectable from the toolbar:

| Persona | Best For |
|---|---|
| ✨ Default | General knowledge, balanced |
| ⚙️ Senior Developer | Architecture, code reviews, production-quality code |
| 💻 Coding Assistant | Write/debug/review code |
| 🔬 Research Assistant | Analysis, citations, web-sourced facts |
| 📊 Data Analyst | Numbers, insights, business intelligence |
| 🎨 Creative Partner | Writing, brainstorming, storytelling |
| ⚡ Concise Mode | Shortest accurate answer, zero fluff |

### 💬 Chat Features
- **Multi-conversation sidebar** with search, rename, delete
- **Persistent history** in localStorage with per-conversation session IDs
- **Markdown rendering** with syntax-highlighted code (highlight.js, 100+ languages)
- **Streaming tokens** with real-time render
- **Regenerate / edit** user messages
- **Export conversation** as Markdown
- **Token stats** per message (tokens generated, tok/s)
- **Drag & drop** files directly into chat
- **Image attachment** for vision-capable models
- **System prompt bar** — per-session custom instructions
- **Parameter controls** — temperature, top-p, max tokens, context length
- **Keyboard shortcuts**: `Ctrl+Enter` send, `Ctrl+K` new chat, `Esc` close modals

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Ollama](https://ollama.com) | AMD: use `ollama-vulkan`; NVIDIA: standard install |
| [ArangoDB](https://arangodb.com) 3.12 | For RAG features |

### 1. Clone and install

```bash
git clone https://github.com/obnoxiousmods/obbyai.git
cd obbyai
uv sync
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env: set ARANGO_PASS and OLLAMA_REMOTE_URL
```

### 3. Pull models

```bash
# Primary model
ollama pull llama3.1:8b

# Required for RAG
ollama pull nomic-embed-text

# Optional extras
ollama pull qwen2.5:7b qwen2.5-coder:7b deepseek-r1:8b
```

### 4. Run

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8091
```

Open **http://localhost:8091**

### Production (systemd)

```ini
[Unit]
Description=ObbyAI Chat Web UI
After=network.target ollama.service arangodb3.service

[Service]
WorkingDirectory=/opt/ai-chat
EnvironmentFile=/opt/ai-chat/.env
ExecStart=/home/user/.local/bin/uv run uvicorn main:app --host 127.0.0.1 --port 8091
Restart=on-failure
User=youruser

[Install]
WantedBy=multi-user.target
```

### HTTPS with nginx

```nginx
server {
    listen 443 ssl;
    server_name ai.yourdomain.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8091;
        proxy_set_header Host $host;
        proxy_buffering off;
    }

    location ~ ^/(api|v1)/ {
        proxy_pass http://127.0.0.1:11434;
    }
}
```

---

## Architecture

```
Browser
  │  HTTPS (nginx SSL termination)
  ▼
Starlette (port 8091)
  ├── GET  /              → index.html (self-contained SPA)
  ├── GET  /servers       → [{id, name, gpu, default_model}]
  ├── GET  /models        → Ollama /api/tags (proxied per server)
  ├── GET  /prompts       → persona list
  ├── POST /upload        → file_processor → RAG ingest
  ├── POST /ingest        → direct text → RAG ingest
  ├── GET  /rag/stats     → document + session message counts
  └── POST /chat          → tool loop → Ollama SSE → frontend SSE
        │
        ├── tools/web_search.py     DuckDuckGo
        ├── tools/rag.py            ArangoDB + nomic-embed-text
        ├── tools/calculator.py     AST eval
        ├── tools/datetime_tool.py  zoneinfo
        └── tools/file_processor.py
              ├── Images   → Gemma4 @ remote:11434
              ├── PDF      → pymupdf4llm
              ├── DOCX     → python-docx
              ├── PPTX     → python-pptx
              ├── XLSX     → openpyxl
              └── Text/Code → chardet + direct decode

ArangoDB (port 8529)
  ├── rag_documents   (chunked text + nomic embeddings)
  └── rag_sessions    (per-conversation message embeddings)

Ollama instances
  ├── Local  127.0.0.1:11434   (RX 580 / Vulkan)
  └── Remote 192.168.1.x:11434 (RTX 2080S / CUDA)
```

### Tool Call Flow

```
User message
    │
    ├─ RAG search (if use_rag=true) → inject top-K chunks into context
    │
    └─ Ollama /api/chat (with tools=[web_search, rag_search, calculator, get_datetime])
            │
            ├─ Model streams tokens → forwarded to browser via SSE
            │
            └─ Model calls tool?
                    │
                    ├─ Execute tool
                    ├─ Stream tool_call + tool_result events to browser
                    └─ Re-submit with tool results → continue streaming
```

---

## Configuration

All configuration is in `.env` (copy from `.env.example`):

```env
ARANGO_URL=http://localhost:8529
ARANGO_USER=root
ARANGO_PASS=your_password
ARANGO_DB=ai_chat_rag

OLLAMA_LOCAL_URL=http://127.0.0.1:11434
OLLAMA_REMOTE_URL=http://192.168.1.x:11434

EMBED_MODEL=nomic-embed-text
VISION_MODEL=gemma4:latest
```

### AMD GPU Notes (RX 580 / Polaris)

The RX 580 2048SP (PCI ID `0x6fdf`) is a Chinese market variant not in ROCm's whitelist. Use Vulkan:

```bash
# Arch Linux
pacman -S ollama-vulkan   # NOT ollama (ROCm-only)

# Verify GPU layers
ollama run llama3.1:8b "hi" 2>&1 | grep -i gpu
# Expected: "offloaded 33/33 layers to GPU"
```

---

## Development

```bash
uv sync
uv run uvicorn main:app --reload --port 8091

# Lint
uv run ruff check .
uv run ruff format .

# Check JS
python3 -c "
import re
html = open('index.html').read()
scripts = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
open('/tmp/check.js','w').write('\n'.join(scripts))
"
node --check /tmp/check.js
```

---

## Roadmap

- [ ] Authentication (nginx basic auth / OAuth)
- [ ] Docker Compose deployment
- [ ] Code execution sandbox (Python REPL tool)
- [ ] Voice input (Whisper via Ollama)
- [ ] Image generation (Stable Diffusion / ComfyUI)
- [ ] Conversation export to HTML
- [ ] Mobile layout improvements
- [ ] Scheduled automation tools

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome.

---

## License

[MIT](LICENSE) © 2026 obnoxiousmods
