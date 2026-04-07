# Contributing to ObbyAI

Thanks for your interest in contributing! This is a personal homelab project but PRs are welcome.

## Development Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Ollama](https://ollama.com) — running locally or on a network machine
- [ArangoDB](https://arangodb.com) 3.12+ — for RAG features
- Node.js 18+ — for JS syntax checking only

### Quick Start

```bash
git clone https://github.com/obnoxiousmods/obbyai.git
cd obbyai

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your ArangoDB password and Ollama URLs

# Pull required models
ollama pull llama3.1:8b        # primary model
ollama pull nomic-embed-text   # for RAG embeddings

# Run development server
uv run uvicorn main:app --reload --port 8091
```

Open http://localhost:8091

## Architecture

```
index.html          Self-contained frontend (HTML+CSS+JS, ~2500 lines)
main.py             Starlette ASGI app — routes, tool loop, streaming
prompts.py          System prompt library (7 personas)
tools/
  web_search.py     DuckDuckGo search (duckduckgo-search)
  rag.py            ArangoDB RAG — ingest, embed, search
  calculator.py     Safe math expression evaluator
  datetime_tool.py  Current date/time
  file_processor.py File ingestion: PDF/DOCX/PPTX/XLSX/images/text
```

## Code Style

- Python: formatted with [ruff](https://docs.astral.sh/ruff/) (`uv run ruff format .`)
- No type stubs required but type hints appreciated on public functions
- JS: vanilla ES2022+, no build step, no frameworks

## Testing

```bash
# Run existing tests
uv run pytest tests/ -v

# Check JS syntax
python3 -c "
import re
html = open('index.html').read()
scripts = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
open('/tmp/check.js','w').write('\n'.join(scripts))
"
node --check /tmp/check.js
```

## Submitting a PR

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes
3. Run lint: `uv run ruff check .`
4. Test manually
5. Push and open a PR against `main`
6. Fill out the PR template

## What's Welcome

- Bug fixes
- New file type support in `tools/file_processor.py`
- Additional Ollama tool specs
- UI improvements to `index.html`
- New personas in `prompts.py`
- Performance improvements to RAG

## What's Out of Scope

- Breaking changes to the single-file frontend approach
- Dependencies that require API keys or external paid services
- Features that compromise the "runs fully offline" design goal
