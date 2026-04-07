"""Web search tool using DuckDuckGo (no API key required)."""
import asyncio
from duckduckgo_search import DDGS


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information, news, documentation, or anything "
            "not in your training data. Use when the user asks about recent events, "
            "wants you to look something up, or when you need up-to-date facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and concise.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-8, default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


async def run(query: str, max_results: int = 5) -> dict:
    max_results = max(1, min(8, max_results))
    try:
        results = await asyncio.to_thread(_search_sync, query, max_results)
        if not results:
            return {"error": "No results found.", "query": query}
        return {
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ],
        }
    except Exception as e:
        return {"error": str(e), "query": query}


def _search_sync(query: str, max_results: int) -> list:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))
