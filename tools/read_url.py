"""read_url tool — fetch and extract text from any URL the model wants to read."""
from __future__ import annotations
import re
import httpx
from bs4 import BeautifulSoup

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "read_url",
        "description": (
            "Fetch and read the full text content of any URL. Use when you need to "
            "read a specific webpage, article, documentation page, or GitHub file in full. "
            "Great for following up on search results you want to read in detail."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters to return (default 5000, max 15000).",
                    "default": 5000,
                },
            },
            "required": ["url"],
        },
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


async def run(url: str, max_chars: int = 5000) -> dict:
    max_chars = max(500, min(15000, max_chars))
    try:
        async with httpx.AsyncClient(
            timeout=12,
            headers=HEADERS,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            ct = resp.headers.get("content-type", "")

            # Plain text / markdown / JSON / code — return raw
            if any(t in ct for t in ("text/plain", "application/json", "text/markdown")):
                return {
                    "url": url,
                    "content_type": ct.split(";")[0].strip(),
                    "content": resp.text[:max_chars],
                }

            if "html" not in ct:
                return {"url": url, "error": f"Unsupported content type: {ct}"}

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "noscript", "iframe", "svg"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title else ""

            main = (
                soup.find("article")
                or soup.find("main")
                or soup.find(id=re.compile(r"content|main|article|readme", re.I))
                or soup.find(class_=re.compile(r"content|main|article|post|entry|readme", re.I))
                or soup.body
            )
            text = re.sub(r"\s{3,}", "\n\n", main.get_text(" ", strip=True)) if main else ""

            return {
                "url":          url,
                "title":        title,
                "content":      text[:max_chars],
                "total_chars":  len(text),
                "truncated":    len(text) > max_chars,
            }

    except httpx.HTTPStatusError as e:
        return {"url": url, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"url": url, "error": str(e)}
