"""python_exec tool — sandboxed Python execution with timeout."""
from __future__ import annotations
import asyncio
import sys
import textwrap

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Execute Python code and return stdout/stderr output. Use for: data analysis, "
            "calculations, string manipulation, parsing, generating charts data, sorting, "
            "working with JSON/CSV data, or any task better solved with code than reasoning. "
            "Has access to: math, json, re, datetime, collections, itertools, statistics, "
            "decimal, fractions, random, string, csv, io, base64, hashlib, urllib.parse."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() for output.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max execution seconds (default 10, max 30).",
                    "default": 10,
                },
            },
            "required": ["code"],
        },
    },
}

# Allowed stdlib modules (no network, no filesystem writes, no subprocess)
_ALLOWED_IMPORTS = {
    "math", "cmath", "decimal", "fractions", "random", "statistics",
    "string", "re", "textwrap", "unicodedata",
    "datetime", "calendar", "time",
    "json", "csv", "io", "base64", "binascii", "struct", "codecs",
    "hashlib", "hmac",
    "collections", "heapq", "bisect", "array", "queue", "itertools", "functools",
    "operator", "copy",
    "pprint", "reprlib",
    "urllib.parse",
    "html", "html.parser",
    "enum", "dataclasses", "typing",
    "abc", "contextlib",
}

_BANNED = [
    "import os", "import sys", "import subprocess", "import socket",
    "import requests", "import httpx", "import urllib.request",
    "__import__", "open(", "exec(", "eval(",
    "importlib", "builtins", "globals(", "locals(",
    "compile(", "getattr", "setattr", "delattr",
]


def _is_safe(code: str) -> tuple[bool, str]:
    lower = code.lower()
    for banned in _BANNED:
        if banned in lower:
            return False, f"Blocked: '{banned}' is not allowed in sandbox."
    return True, ""


async def run(code: str, timeout: int = 10) -> dict:
    timeout = max(1, min(30, timeout))
    safe, reason = _is_safe(code)
    if not safe:
        return {"error": reason, "code": code[:200]}

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_exec_sync, code),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        return {"error": f"Execution timed out after {timeout}s", "code": code[:200]}
    except Exception as e:
        return {"error": str(e), "code": code[:200]}


def _exec_sync(code: str) -> dict:
    import io, contextlib

    # Build a restricted globals dict
    safe_globals: dict = {"__builtins__": _safe_builtins()}

    # Pre-import allowed modules into the namespace
    for mod in _ALLOWED_IMPORTS:
        try:
            top = mod.split(".")[0]
            import importlib
            safe_globals[top] = importlib.import_module(top)
        except ImportError:
            pass

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    error = None

    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            exec(textwrap.dedent(code), safe_globals)  # noqa: S102
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

    out = stdout_buf.getvalue().strip()
    err = stderr_buf.getvalue().strip()
    if error:
        err = (err + "\n" + error).strip()

    return {
        "stdout": out[:4000] if out else "",
        "stderr": err[:1000] if err else "",
        "success": error is None,
    }


def _safe_builtins() -> dict:
    safe_names = [
        "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr",
        "dict", "dir", "divmod", "enumerate", "filter", "float", "format",
        "frozenset", "hasattr", "hash", "hex", "id", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max",
        "min", "next", "object", "oct", "ord", "pow", "print", "range",
        "repr", "reversed", "round", "set", "slice", "sorted", "str",
        "sum", "super", "tuple", "type", "vars", "zip",
        "True", "False", "None",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
        "ArithmeticError", "ZeroDivisionError", "OverflowError",
    ]
    import builtins, importlib as _il
    result = {name: getattr(builtins, name) for name in safe_names if hasattr(builtins, name)}

    # Safe __import__ restricted to allowlist
    def _safe_import(name, *args, **kwargs):
        base = name.split(".")[0]
        if base in _ALLOWED_IMPORTS:
            return _il.import_module(name)
        raise ImportError(f"'{name}' is not available in the sandbox.")
    result["__import__"] = _safe_import
    return result
