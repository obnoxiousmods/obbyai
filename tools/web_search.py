"""
Web search tool — multi-engine with content fetching and caching.

Engines (in priority order, run in parallel):
  1. DuckDuckGo  (ddgs)
  2. Wikipedia   (REST API, no key)
  3. Brave       (free tier API key optional via env BRAVE_API_KEY)

Top result pages are fetched and their main text extracted so the model
gets real content, not just snippets.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[float, dict]] = {}   # key → (timestamp, result)
_CACHE_TTL = 300                              # 5 minutes

# ── Config ────────────────────────────────────────────────────────────────────
BRAVE_API_KEY   = os.getenv("BRAVE_API_KEY", "")
FETCH_TIMEOUT   = 6          # seconds per page fetch
FETCH_MAX_CHARS = 2000       # chars to extract from fetched pages
MAX_FETCH_URLS  = 2          # how many top URLs to fetch content from
DDG_MAX         = 6
WIKI_MAX        = 2

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Tool spec ─────────────────────────────────────────────────────────────────
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information, news, software releases, "
            "documentation, prices, or anything not in your training data. "
            "Returns titles, URLs, snippets, AND full page content from top results. "
            "Use whenever the user asks about recent events, wants a look-up, or "
            "when you need up-to-date or verified facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific, targeted search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Results to return (1-10, default 6).",
                    "default": 6,
                },
                "fetch_content": {
                    "type": "boolean",
                    "description": "Fetch and include full text from top URLs (default true).",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
}


# ── Main entry ─────────────────────────────────────────────────────────────────
async def run(query: str, max_results: int = 6, fetch_content: bool = True) -> dict:
    max_results = max(1, min(10, max_results))
    cache_key   = hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()

    # Cache hit
    if cache_key in _CACHE:
        ts, cached = _CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            cached["cached"] = True
            return cached

    try:
        # Run DDG + Wikipedia in parallel
        ddg_task  = asyncio.create_task(_ddg_search(query, DDG_MAX))
        wiki_task = asyncio.create_task(_wiki_search(query, WIKI_MAX))
        brave_task = asyncio.create_task(_brave_search(query, 4)) if BRAVE_API_KEY else None

        tasks = [t for t in [ddg_task, wiki_task, brave_task] if t]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge, deduplicate by URL
        merged: list[dict] = []
        seen_urls: set[str] = set()
        for lst in results_lists:
            if isinstance(lst, Exception) or not lst:
                continue
            for r in lst:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged.append(r)

        if not merged:
            return {"error": "No results found across all search engines.", "query": query}

        merged = merged[:max_results]

        # Fetch page content for top URLs
        if fetch_content:
            urls_to_fetch = [
                r["url"] for r in merged[:MAX_FETCH_URLS]
                if r.get("url") and not r.get("content")
            ]
            if urls_to_fetch:
                fetched = await asyncio.gather(
                    *[_fetch_page_text(u) for u in urls_to_fetch],
                    return_exceptions=True,
                )
                for i, text in enumerate(fetched):
                    if isinstance(text, str) and text:
                        merged[i]["content"] = text

        result = {
            "query":   query,
            "count":   len(merged),
            "results": merged,
            "cached":  False,
        }
        _CACHE[cache_key] = (time.time(), result)
        return result

    except Exception as e:
        return {"error": str(e), "query": query}


# ── DuckDuckGo ────────────────────────────────────────────────────────────────
async def _ddg_search(query: str, n: int) -> list[dict]:
    try:
        raw = await asyncio.to_thread(_ddg_sync, query, n)
        return [
            {
                "source":  "ddg",
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    except Exception:
        return []


def _ddg_sync(query: str, n: int) -> list:
    with DDGS() as d:
        return list(d.text(query, max_results=n))


# ── Wikipedia ─────────────────────────────────────────────────────────────────
async def _wiki_search(query: str, n: int) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=8, headers=HEADERS) as client:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action":   "query",
                    "list":     "search",
                    "srsearch": query,
                    "srlimit":  n,
                    "format":   "json",
                    "utf8":     1,
                },
            )
            data = resp.json()
            items = data.get("query", {}).get("search", [])
            return [
                {
                    "source":  "wikipedia",
                    "title":   it["title"],
                    "url":     f"https://en.wikipedia.org/wiki/{it['title'].replace(' ', '_')}",
                    "snippet": re.sub(r"<[^>]+>", "", it.get("snippet", "")),
                }
                for it in items
            ]
    except Exception:
        return []


# ── Brave Search (optional) ───────────────────────────────────────────────────
async def _brave_search(query: str, n: int) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=8, headers=HEADERS) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": n},
                headers={**HEADERS, "Accept": "application/json",
                         "X-Subscription-Token": BRAVE_API_KEY},
            )
            items = resp.json().get("web", {}).get("results", [])
            return [
                {
                    "source":  "brave",
                    "title":   it.get("title", ""),
                    "url":     it.get("url", ""),
                    "snippet": it.get("description", ""),
                }
                for it in items
            ]
    except Exception:
        return []


# ── Page content fetcher ──────────────────────────────────────────────────────
async def _fetch_page_text(url: str) -> str:
    """Fetch a URL and extract clean main-body text."""
    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            headers=HEADERS,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            resp = await client.get(url)
            ct = resp.headers.get("content-type", "")
            if "html" not in ct:
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "noscript", "iframe"]):
                tag.decompose()

            # Prefer article/main content
            main = (
                soup.find("article")
                or soup.find("main")
                or soup.find(id=re.compile(r"content|main|article", re.I))
                or soup.find(class_=re.compile(r"content|main|article|post|entry", re.I))
                or soup.body
            )
            if not main:
                return ""

            text = re.sub(r"\s{3,}", "\n\n", main.get_text(" ", strip=True))
            return text[:FETCH_MAX_CHARS].strip()
    except Exception:
        return ""

