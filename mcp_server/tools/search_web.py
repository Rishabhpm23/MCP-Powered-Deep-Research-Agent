"""
search_web.py
─────────────
MCP Tool: search_web
Uses the Tavily Search API to retrieve ranked, relevant web results for a query.
Returns a structured list of results with titles, URLs, snippets, and scores.
"""

import os
import logging
from typing import Any

from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def search_web(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> dict[str, Any]:
    """
    Search the web using the Tavily API.

    Args:
        query:        The research query string.
        max_results:  Maximum number of results to return (default 5).
        search_depth: 'basic' or 'advanced' (advanced fetches more detail).

    Returns:
        A dict with keys:
          - query (str)
          - results (list[dict])  — each has title, url, content, score
          - total_results (int)
          - error (str | None)
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY not set in environment.")
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "error": "TAVILY_API_KEY is not configured.",
        }

    try:
        client = TavilyClient(api_key=api_key)
        logger.info(f"[search_web] Searching: '{query}' | depth={search_depth} | max={max_results}")

        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )

        results = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": round(item.get("score", 0.0), 4),
                }
            )

        logger.info(f"[search_web] Found {len(results)} results.")
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "error": None,
        }

    except Exception as e:
        logger.exception(f"[search_web] Error: {e}")
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "error": str(e),
        }
