"""
ObbyAI — local AI chat backend
Starlette + Ollama + tools (web search, RAG, calculator, datetime) + file processing
"""
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

# Load .env before anything else
def _load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
_load_env()

import httpx
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import HTMLResponse, StreamingResponse, JSONResponse
from starlette.routing import Route

from prompts import get as get_prompt, list_prompts
from tools import web_search, calculator, datetime_tool
from tools import rag
from tools.file_processor import process_file, supported_extensions, ProcessingError, MODE_AUTO, MODE_TEXT, MODE_VISION, MODE_BOTH

# ── Config ─────────────────────────────────────────────────────────────────────

HTML_FILE = Path(__file__).parent / "index.html"

SERVERS = {
    "local": {
        "url":           os.getenv("OLLAMA_LOCAL_URL",  "http://127.0.0.1:11434"),
        "name":          "RX 580",
        "gpu":           "AMD RX 580 8GB",
        "default_model": "llama3.1:8b",
    },
    "remote": {
        "url":           os.getenv("OLLAMA_REMOTE_URL", "http://192.168.1.220:11434"),
        "name":          "RTX 2080 Super",
        "gpu":           "NVIDIA RTX 2080 Super 8GB",
        "default_model": "gemma4:latest",
    },
}
DEFAULT_SERVER = "remote"
DEFAULT_MODEL  = "gemma4:latest"

TOOL_CAPABLE_MODELS = {
    "llama3.1:8b", "llama3.2:1b", "llama3.3:70b",
    "qwen2.5:7b", "qwen2.5-coder:7b", "qwen3:8b",
    "mistral:7b-instruct", "mistral-nemo:12b",
    "phi4-mini:3.8b",
}

TOOL_SPECS = [
    web_search.TOOL_SPEC,
    rag.TOOL_SPEC,
    calculator.TOOL_SPEC,
    datetime_tool.TOOL_SPEC,
]

# ── Basic routes ───────────────────────────────────────────────────────────────

async def index(request: Request):
    return HTMLResponse(HTML_FILE.read_text())

async def get_servers(request: Request):
    return JSONResponse([
        {"id": sid, **{k: v for k, v in s.items() if k != "url"}}
        for sid, s in SERVERS.items()
    ])

async def get_models(request: Request):
    server_id = request.query_params.get("server", DEFAULT_SERVER)
    server = SERVERS.get(server_id, SERVERS[DEFAULT_SERVER])
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{server['url']}/api/tags")
            models_data = r.json().get("models", [])
            return JSONResponse([
                {"name": m["name"], "size": m.get("size", 0), "details": m.get("details", {})}
                for m in models_data
            ])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

async def get_prompts(request: Request):
    return JSONResponse(list_prompts())

async def rag_stats(request: Request):
    try:
        db = rag.get_db()
        return JSONResponse({
            "documents":        db.collection("rag_documents").count(),
            "session_messages": db.collection("rag_sessions").count(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def get_extensions(request: Request):
    return JSONResponse(supported_extensions())

# ── File upload & ingest ───────────────────────────────────────────────────────

async def upload(request: Request):
    try:
        form      = await request.form()
        file: UploadFile = form.get("file")
        session_id = str(form.get("session_id", ""))
        mode       = str(form.get("mode", MODE_AUTO))
        ingest_rag = str(form.get("ingest", "true")).lower() != "false"

        # Validate mode
        if mode not in (MODE_AUTO, MODE_TEXT, MODE_VISION, MODE_BOTH):
            mode = MODE_AUTO

        if not file:
            return JSONResponse({"error": "No file provided"}, status_code=400)

        filename = file.filename or "upload"
        data     = await file.read()

        if not data:
            return JSONResponse({"error": "Empty file"}, status_code=400)

        # Process the file with the requested mode
        result = await process_file(filename, data, mode=mode)

        text = result.get("text", "").strip()
        rag_result = None
        if text and ingest_rag:
            rag_result = await rag.ingest_document(text, source=filename, session_id=session_id)

        return JSONResponse({
            "filename":           result["filename"],
            "type":               result["type"],
            "ext":                result["ext"],
            "mode_used":          result.get("mode_used", mode),
            "size_bytes":         result["size_bytes"],
            "pages":              result.get("pages"),
            "truncated":          result.get("truncated", False),
            "text_preview":       text[:500] if text else "",
            "text_extracted":     result.get("text_extracted"),
            "vision_description": result.get("vision_description"),
            "rag_result":         rag_result,
            "ingested":           ingest_rag and rag_result is not None,
        })

    except ProcessingError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": f"Upload failed: {e}"}, status_code=500)

async def ingest(request: Request):
    """Direct text ingest (no file)."""
    try:
        body       = await request.json()
        text       = body.get("text", "").strip()
        source     = body.get("source", "paste")
        session_id = body.get("session_id", "")
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        result = await rag.ingest_document(text, source, session_id)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Tool execution ─────────────────────────────────────────────────────────────

async def execute_tool(name: str, args: dict, session_id: str) -> str:
    try:
        if name == "web_search":
            result = await web_search.run(
                query=args.get("query", ""),
                max_results=args.get("max_results", 5),
            )
        elif name == "rag_search":
            result = await rag.run_tool(query=args.get("query", ""), session_id=session_id)
        elif name == "calculator":
            result = await calculator.run(expression=args.get("expression", ""))
        elif name == "get_datetime":
            result = await datetime_tool.run(timezone_name=args.get("timezone", "local"))
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

# ── Chat ────────────────────────────────────────────────────────────────────────

async def chat(request: Request):
    body       = await request.json()
    messages   = list(body.get("messages", []))
    model      = body.get("model", DEFAULT_MODEL)
    server_id  = body.get("server", DEFAULT_SERVER)
    session_id = body.get("session_id", str(uuid.uuid4()))
    prompt_id  = body.get("prompt_id", "default")
    use_tools  = body.get("use_tools", True)
    use_rag    = body.get("use_rag", True)

    server     = SERVERS.get(server_id, SERVERS[DEFAULT_SERVER])
    ollama_url = server["url"]

    options = {k: v for k, v in {
        "temperature":  body.get("temperature"),
        "top_p":        body.get("top_p"),
        "num_predict":  body.get("max_tokens"),
        "num_ctx":      body.get("context_length"),
    }.items() if v is not None}

    # System prompt
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": get_prompt(prompt_id)})

    # RAG context injection
    rag_context = []
    if use_rag:
        user_texts = [m.get("content","") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
        if user_texts:
            try:
                rag_context = await rag.search(user_texts[-1], session_id)
            except Exception:
                pass

    if rag_context:
        ctx_block = "\n\n".join(
            f"[Source: {r.get('source','?')} | score: {r.get('score',0):.2f}]\n{r['text']}"
            for r in rag_context
        )
        messages.insert(-1, {
            "role": "system",
            "content": f"Relevant context from knowledge base:\n\n{ctx_block}",
        })

    supports_tools = use_tools and any(
        model == m or model.startswith(m.split(":")[0] + ":")
        for m in TOOL_CAPABLE_MODELS
    )

    # Store user message async
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if user_msgs:
        last_content = user_msgs[-1].get("content", "")
        if isinstance(last_content, str) and last_content:
            asyncio.create_task(rag.store_message("user", last_content, session_id))

    async def stream():
        current_messages  = list(messages)
        tool_iterations   = 0
        max_tool_iters    = 5

        while True:
            payload = {
                "model":    model,
                "messages": current_messages,
                "stream":   True,
                "options":  options,
            }
            if supports_tools and tool_iterations < max_tool_iters:
                payload["tools"] = TOOL_SPECS

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{ollama_url}/api/chat", json=payload) as resp:
                        full_content = ""
                        tool_calls   = []

                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                            except Exception:
                                continue

                            msg    = chunk.get("message", {})
                            token  = msg.get("content", "")
                            tcalls = msg.get("tool_calls", [])
                            done   = chunk.get("done", False)

                            if token:
                                full_content += token
                                yield f"data: {json.dumps({'t': token})}\n\n"

                            if tcalls:
                                tool_calls.extend(tcalls)

                            if done:
                                if not tool_calls:
                                    asyncio.create_task(
                                        rag.store_message("assistant", full_content, session_id)
                                    )
                                    yield f"data: {json.dumps({'done': True, 'eval_count': chunk.get('eval_count',0), 'eval_duration': chunk.get('eval_duration',0), 'prompt_eval_count': chunk.get('prompt_eval_count',0), 'rag_context_count': len(rag_context)})}\n\n"
                                    return
                                break

                if tool_calls:
                    tool_iterations += 1
                    current_messages.append({
                        "role": "assistant", "content": full_content, "tool_calls": tool_calls,
                    })
                    for tc in tool_calls:
                        fn   = tc.get("function", {})
                        name = fn.get("name", "")
                        args_raw = fn.get("arguments", {})
                        args = args_raw if isinstance(args_raw, dict) else json.loads(args_raw)

                        yield f"data: {json.dumps({'tool_call': {'name': name, 'args': args}})}\n\n"
                        result_str = await execute_tool(name, args, session_id)
                        yield f"data: {json.dumps({'tool_result': {'name': name, 'result': json.loads(result_str)}})}\n\n"

                        current_messages.append({"role": "tool", "content": result_str})
                else:
                    break

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── App ─────────────────────────────────────────────────────────────────────────

app = Starlette(routes=[
    Route("/",           index),
    Route("/servers",    get_servers),
    Route("/models",     get_models),
    Route("/prompts",    get_prompts),
    Route("/extensions", get_extensions),
    Route("/ingest",     ingest,  methods=["POST"]),
    Route("/upload",     upload,  methods=["POST"]),
    Route("/rag/stats",  rag_stats),
    Route("/chat",       chat,    methods=["POST"]),
])
