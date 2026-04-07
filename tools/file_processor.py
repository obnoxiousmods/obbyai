"""
File processor — handles all supported upload types.

Image types  → Gemma4 vision on 192.168.1.220 for description/OCR
PDF          → pymupdf4llm → markdown
DOCX/DOC     → python-docx → plain text
PPTX         → python-pptx → slide text
XLSX/XLS     → openpyxl → CSV-style text
TXT/MD/JSON/YAML/TOML/XML/CSV/code → direct UTF-8 decode
"""
import asyncio
import base64
import io
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Optional

import chardet
import httpx

# ── Constants ──────────────────────────────────────────────────────────────────

VISION_URL   = os.getenv("OLLAMA_REMOTE_URL", "http://192.168.1.220:11434")
VISION_MODEL = os.getenv("VISION_MODEL",      "gemma4:latest")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg"}
PDF_EXTS   = {".pdf"}
WORD_EXTS  = {".docx", ".doc"}
PPT_EXTS   = {".pptx", ".ppt"}
XL_EXTS    = {".xlsx", ".xls", ".ods"}
TEXT_EXTS  = {
    ".txt", ".md", ".markdown", ".rst",
    ".json", ".jsonl", ".ndjson",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".html", ".htm", ".svg",
    ".csv", ".tsv",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".rs", ".go", ".java", ".kt", ".swift", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".rb", ".php", ".lua", ".r", ".jl", ".ex", ".exs", ".elm",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".proto",
    ".css", ".scss", ".sass", ".less",
    ".log", ".env", ".gitignore", ".dockerignore", ".editorconfig",
    ".Dockerfile", ".makefile", ".mk",
    ".ipynb",
}
MAX_TEXT_BYTES = 5 * 1024 * 1024   # 5 MB text limit per file
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB image limit


class ProcessingError(Exception):
    pass


# ── Dispatcher ──────────────────────────────────────────────────────────────────

async def process_file(filename: str, data: bytes) -> dict:
    """
    Main entry point. Returns:
    {
        "filename": str,
        "ext": str,
        "type": str,           # "image" | "pdf" | "word" | "pptx" | "excel" | "text"
        "text": str,           # extracted text for RAG
        "vision_description": str | None,  # set for images
        "pages": int | None,
        "size_bytes": int,
        "truncated": bool,
    }
    """
    ext = Path(filename).suffix.lower()
    if not ext and "." not in filename:
        # Try to detect from content
        ext = _sniff_ext(data)

    size = len(data)
    result = {
        "filename": filename,
        "ext": ext,
        "size_bytes": size,
        "truncated": False,
        "vision_description": None,
        "pages": None,
    }

    if ext in IMAGE_EXTS or _is_image(data):
        result["type"] = "image"
        desc = await _process_image(filename, data)
        result["text"] = desc
        result["vision_description"] = desc

    elif ext in PDF_EXTS:
        result["type"] = "pdf"
        text, pages = _process_pdf(data)
        result["text"] = text
        result["pages"] = pages

    elif ext in WORD_EXTS:
        result["type"] = "word"
        result["text"] = _process_docx(data, ext)

    elif ext in PPT_EXTS:
        result["type"] = "pptx"
        text, pages = _process_pptx(data)
        result["text"] = text
        result["pages"] = pages

    elif ext in XL_EXTS:
        result["type"] = "excel"
        text, pages = _process_excel(data, ext)
        result["text"] = text
        result["pages"] = pages  # sheets

    else:
        # Default: treat as text
        result["type"] = "text"
        result["text"] = _decode_text(data, filename)

    # Truncate if too large
    if len(result.get("text", "")) > MAX_TEXT_BYTES:
        result["text"] = result["text"][:MAX_TEXT_BYTES]
        result["truncated"] = True

    return result


# ── Image → Gemma4 vision ──────────────────────────────────────────────────────

async def _process_image(filename: str, data: bytes) -> str:
    if len(data) > MAX_IMAGE_BYTES:
        raise ProcessingError(f"Image too large: {len(data) / 1e6:.1f} MB (max 20 MB)")

    b64 = base64.b64encode(data).decode()
    ext = Path(filename).suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "tif": "tiff"}
    mime_ext = mime_map.get(ext, ext) or "jpeg"

    prompt = (
        "Analyze this image thoroughly. Do ALL of the following:\n"
        "1. Extract ALL visible text exactly as written (OCR).\n"
        "2. Describe the image content, layout, and context.\n"
        "3. If it contains data, tables, charts, or diagrams — describe their content and values.\n"
        "4. Note any code, URLs, emails, or structured data visible.\n"
        "5. Describe colors, style, and visual design if relevant.\n"
        "Format your response in clear sections."
    )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{VISION_URL}/api/chat",
                json={
                    "model": VISION_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                        "images": [b64],
                    }],
                    "stream": False,
                },
            )
            data_out = r.json()
            return data_out.get("message", {}).get("content", "").strip()
    except Exception as e:
        raise ProcessingError(f"Vision analysis failed: {e}")


# ── PDF ──────────────────────────────────────────────────────────────────────

def _process_pdf(data: bytes) -> tuple[str, int]:
    try:
        import pymupdf4llm
        import fitz  # pymupdf

        doc = fitz.open(stream=data, filetype="pdf")
        pages = doc.page_count

        # Use pymupdf4llm for markdown output (preserves tables, layout)
        md_text = pymupdf4llm.to_markdown(doc)

        if not md_text.strip():
            # Fallback: plain text extraction
            parts = []
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    parts.append(f"## Page {i+1}\n\n{text}")
            md_text = "\n\n".join(parts)

        doc.close()
        return md_text, pages
    except Exception as e:
        raise ProcessingError(f"PDF extraction failed: {e}")


# ── DOCX/DOC ──────────────────────────────────────────────────────────────────

def _process_docx(data: bytes, ext: str) -> str:
    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            parts = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    style = para.style.name if para.style else ""
                    if "Heading" in style:
                        level = re.search(r"\d+", style)
                        hashes = "#" * (int(level.group()) if level else 1)
                        parts.append(f"{hashes} {text}")
                    else:
                        parts.append(text)
            # Also extract tables
            for table in doc.tables:
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append(" | ".join(cells))
                if rows:
                    parts.append("\n".join(rows))
            return "\n\n".join(parts)
        except Exception as e:
            raise ProcessingError(f"DOCX extraction failed: {e}")
    else:
        # .doc — try extracting as raw text
        try:
            text = _decode_text(data, "file.doc")
            # Clean up binary garbage
            text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)
            return text
        except Exception as e:
            raise ProcessingError(f"DOC extraction failed: {e}")


# ── PPTX ──────────────────────────────────────────────────────────────────────

def _process_pptx(data: bytes) -> tuple[str, int]:
    try:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(data))
        parts = []
        for i, slide in enumerate(prs.slides):
            slide_parts = [f"## Slide {i+1}"]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_parts.append(shape.text.strip())
                # Extract table text
                if shape.has_table:
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        slide_parts.append(" | ".join(cells))
            parts.append("\n\n".join(slide_parts))
        return "\n\n---\n\n".join(parts), len(prs.slides)
    except Exception as e:
        raise ProcessingError(f"PPTX extraction failed: {e}")


# ── Excel/ODS ──────────────────────────────────────────────────────────────────

def _process_excel(data: bytes, ext: str) -> tuple[str, int]:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        parts = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                # Skip entirely empty rows
                if any(c.strip() for c in cells):
                    rows.append(" | ".join(cells))
            if rows:
                parts.append(f"## Sheet: {sheet_name}\n\n" + "\n".join(rows))
        return "\n\n".join(parts), len(wb.sheetnames)
    except Exception as e:
        raise ProcessingError(f"Excel extraction failed: {e}")


# ── Text / code / structured ──────────────────────────────────────────────────

def _decode_text(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    # Try UTF-8 first
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # Use chardet to detect encoding
        detected = chardet.detect(data)
        encoding = detected.get("encoding") or "latin-1"
        try:
            text = data.decode(encoding)
        except Exception:
            text = data.decode("latin-1", errors="replace")

    # For JSON: pretty-print if compact
    if ext in (".json", ".jsonl", ".ndjson"):
        try:
            if ext == ".json":
                obj = json.loads(text)
                text = json.dumps(obj, indent=2, ensure_ascii=False)
            else:
                # JSONL: parse each line
                lines = []
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            lines.append(json.dumps(json.loads(line), indent=2))
                        except Exception:
                            lines.append(line)
                text = "\n\n".join(lines)
        except Exception:
            pass  # Keep as-is

    return text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_image(data: bytes) -> bool:
    signatures = [
        b"\xff\xd8\xff",        # JPEG
        b"\x89PNG\r\n\x1a\n",  # PNG
        b"GIF87a", b"GIF89a",   # GIF
        b"RIFF",                # WebP (RIFF....WEBP)
        b"BM",                  # BMP
        b"II*\x00", b"MM\x00*", # TIFF
    ]
    for sig in signatures:
        if data[:len(sig)] == sig:
            return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def _sniff_ext(data: bytes) -> str:
    if _is_image(data):
        if data[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        return ".jpg"
    if data[:4] == b"%PDF":
        return ".pdf"
    if data[:2] in (b"PK",):
        return ".docx"  # Could be xlsx/pptx too, but default to docx
    return ".txt"


def supported_extensions() -> list[str]:
    return sorted(IMAGE_EXTS | PDF_EXTS | WORD_EXTS | PPT_EXTS | XL_EXTS | TEXT_EXTS)
