# Changelog

All notable changes to ObbyAI are documented here.

## [Unreleased]

### Added
- Multi-GPU server routing (local RX 580 + remote RTX 2080 Super)
- RAG knowledge base powered by ArangoDB + nomic-embed-text embeddings
- Tool calling: web search (DuckDuckGo), calculator, datetime, RAG search
- Comprehensive file ingestion: PDF, DOCX, PPTX, XLSX, images, 30+ text/code formats
- Image vision analysis via Gemma4 on RTX 2080 Super
- Persona/prompt library: 7 curated system prompts
- Per-conversation session IDs for isolated RAG context
- pymupdf4llm for high-fidelity PDF → markdown extraction

### Changed
- Default model changed from gemma4 to llama3.1:8b (GPU-stable on RX 580 Vulkan)
- Switched from ROCm to Vulkan backend for AMD RX 580 compatibility
- Backend refactored into modular tools/ package

### Fixed
- JavaScript syntax error in DEFAULT_SETTINGS (unescaped apostrophe)
- Gemma4 corrupted output on Vulkan (multimodal model, 9.6GB VRAM overflow)
- Short-prompt looping bug root-caused to tokenizer mismatch in gemma4 GGUF
