"""
Web search tool — multi-engine with content fetching, caching, and structured
source APIs (GitHub releases, PyPI, npm).

Engines run in parallel:
  1. DuckDuckGo  (ddgs)
  2. Wikipedia   (REST API)
  3. Brave       (optional, env BRAVE_API_KEY)

After results are gathered, GitHub repo URLs are auto-detected and queried
via the GitHub Releases API for exact version data. PyPI and npm are checked
when package-like queries are detected.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 300   # 5 minutes

# ── Config ────────────────────────────────────────────────────────────────────
BRAVE_API_KEY  = os.getenv("BRAVE_API_KEY", "")
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "")
FETCH_TIMEOUT  = 8
FETCH_MAX_CHARS = 3000
MAX_FETCH_URLS  = 30
DDG_MAX         = 20
WIKI_MAX        = 4

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}

# ── Patterns ──────────────────────────────────────────────────────────────────
_GH_URL_RE   = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
_PYPI_RE     = re.compile(r"pypi\.org/project/([A-Za-z0-9_.-]+)")
_NPM_RE      = re.compile(r"npmjs\.com/package/([A-Za-z0-9@/_.-]+)")

# ── Tool spec ─────────────────────────────────────────────────────────────────
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information, software versions, release notes, "
            "news, documentation, prices, or anything not in your training data. "
            "Automatically queries GitHub Releases API, PyPI, and npm for exact version "
            "data when relevant. Returns titles, URLs, full page content, AND structured "
            "release/package info. Use whenever the user asks about versions, recent events, "
            "documentation, or needs verified up-to-date facts."
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
                    "description": "Web results to return (5-30, default 10).",
                    "default": 10,
                },
                "fetch_content": {
                    "type": "boolean",
                    "description": "Fetch full page text from all result URLs (default true).",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
}


# ── Main entry ─────────────────────────────────────────────────────────────────
async def run(query: str, max_results: int = 10, fetch_content: bool = True) -> dict:
    max_results = max(5, min(30, max_results))
    cache_key   = hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()

    if cache_key in _CACHE:
        ts, cached = _CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            cached["cached"] = True
            return cached

    try:
        # ── Search engines in parallel ────────────────────────────────────────
        ddg_task   = asyncio.create_task(_ddg_search(query, max(max_results, DDG_MAX)))
        wiki_task  = asyncio.create_task(_wiki_search(query, WIKI_MAX))
        brave_task = asyncio.create_task(_brave_search(query, 10)) if BRAVE_API_KEY else None

        engine_tasks = [t for t in [ddg_task, wiki_task, brave_task] if t]
        results_lists = await asyncio.gather(*engine_tasks, return_exceptions=True)

        # Merge + deduplicate
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
            return {"error": "No results found.", "query": query}

        merged = merged[:max_results]

        # ── Structured source APIs ─────────────────────────────────────────────
        # Collect all URLs + snippets to scan for known sources
        all_text = " ".join(
            r.get("url", "") + " " + r.get("snippet", "") for r in merged
        )
        structured = await _fetch_structured_sources(query, all_text, merged)

        # ── Page content fetch (all results in parallel) ───────────────────────
        if fetch_content:
            fetch_coros = [_fetch_page_text(r["url"]) for r in merged if r.get("url")]
            fetched = await asyncio.gather(*fetch_coros, return_exceptions=True)
            for i, text in enumerate(fetched):
                if isinstance(text, str) and text:
                    merged[i]["content"] = text

        result: dict = {
            "query":   query,
            "count":   len(merged),
            "results": merged,
            "cached":  False,
        }
        if structured:
            result["structured_sources"] = structured

        _CACHE[cache_key] = (time.time(), result)
        return result

    except Exception as e:
        return {"error": str(e), "query": query}


# ── Structured source fetchers ─────────────────────────────────────────────────
async def _fetch_structured_sources(query: str, all_text: str, results: list[dict]) -> list[dict]:
    """Auto-detect GitHub repos, PyPI packages, npm packages and fetch structured data."""
    tasks = []

    # GitHub: collect unique owner/repo pairs from all result URLs
    gh_seen: set[str] = set()
    for r in results:
        url = r.get("url", "")
        m = _GH_URL_RE.search(url)
        if m:
            key = f"{m.group(1)}/{m.group(2)}"
            # Skip noise repos (forks named same as original that show up as clones)
            if key not in gh_seen and not _is_noise_repo(m.group(2)):
                gh_seen.add(key)
                tasks.append(asyncio.create_task(_github_release_info(m.group(1), m.group(2))))

    # PyPI
    pypi_seen: set[str] = set()
    for m in _PYPI_RE.finditer(all_text):
        pkg = m.group(1)
        if pkg not in pypi_seen:
            pypi_seen.add(pkg)
            tasks.append(asyncio.create_task(_pypi_info(pkg)))

    # npm
    npm_seen: set[str] = set()
    for m in _NPM_RE.finditer(all_text):
        pkg = m.group(1).split("/")[0]  # strip sub-paths
        if pkg not in npm_seen:
            npm_seen.add(pkg)
            tasks.append(asyncio.create_task(_npm_info(pkg)))

    if not tasks:
        return []

    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in gathered if isinstance(r, dict) and r]


def _is_noise_repo(name: str) -> bool:
    """Filter out non-software GitHub repos."""
    noise = {".github", "dotfiles", "awesome-list", "awesome", "homebrew-cask"}
    return name.lower() in noise


# ── GitHub Releases API ────────────────────────────────────────────────────────
async def _github_release_info(owner: str, repo: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=8, headers={**HEADERS, **GH_HEADERS}) as client:
            # Latest release
            r = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
            )
            if r.status_code == 404:
                # Try tags instead
                t = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/tags",
                    params={"per_page": 1},
                )
                if t.status_code == 200:
                    tags = t.json()
                    if tags:
                        return {
                            "source": "github_tags",
                            "repo":   f"{owner}/{repo}",
                            "url":    f"https://github.com/{owner}/{repo}/tags",
                            "latest_tag": tags[0]["name"],
                        }
                return None
            if r.status_code != 200:
                return None

            d = r.json()

            # Also grab last 5 releases for context
            releases_r = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/releases",
                params={"per_page": 5},
            )
            recent = []
            if releases_r.status_code == 200:
                for rel in releases_r.json():
                    recent.append({
                        "version":      rel.get("tag_name", ""),
                        "name":         rel.get("name", ""),
                        "published_at": rel.get("published_at", "")[:10],
                        "prerelease":   rel.get("prerelease", False),
                    })

            # Truncate release body to avoid overwhelming the model
            body = (d.get("body") or "").strip()
            if len(body) > 1500:
                body = body[:1500] + "\n…(truncated)"

            return {
                "source":          "github_releases",
                "repo":            f"{owner}/{repo}",
                "url":             d.get("html_url", f"https://github.com/{owner}/{repo}/releases"),
                "latest_version":  d.get("tag_name", ""),
                "release_name":    d.get("name", ""),
                "published_at":    (d.get("published_at") or "")[:10],
                "prerelease":      d.get("prerelease", False),
                "release_notes":   body,
                "recent_releases": recent,
            }
    except Exception:
        return None


# ── PyPI ──────────────────────────────────────────────────────────────────────
async def _pypi_info(package: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=6, headers=HEADERS) as client:
            r = await client.get(f"https://pypi.org/pypi/{package}/json")
            if r.status_code != 200:
                return None
            d    = r.json()
            info = d.get("info", {})
            return {
                "source":          "pypi",
                "package":         info.get("name", package),
                "url":             f"https://pypi.org/project/{package}/",
                "latest_version":  info.get("version", ""),
                "summary":         info.get("summary", ""),
                "home_page":       info.get("home_page") or info.get("project_url", ""),
                "requires_python": info.get("requires_python", ""),
            }
    except Exception:
        return None


# ── npm ───────────────────────────────────────────────────────────────────────
async def _npm_info(package: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=6, headers=HEADERS) as client:
            r = await client.get(f"https://registry.npmjs.org/{package}/latest")
            if r.status_code != 200:
                return None
            d = r.json()
            return {
                "source":         "npm",
                "package":        d.get("name", package),
                "url":            f"https://www.npmjs.com/package/{package}",
                "latest_version": d.get("version", ""),
                "description":    d.get("description", ""),
                "license":        d.get("license", ""),
            }
    except Exception:
        return None


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
            items = resp.json().get("query", {}).get("search", [])
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


# ── Brave Search ──────────────────────────────────────────────────────────────
async def _brave_search(query: str, n: int) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": n},
                headers={
                    "Accept":               "application/json",
                    "X-Subscription-Token": BRAVE_API_KEY,
                },
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
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "noscript", "iframe"]):
                tag.decompose()
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

