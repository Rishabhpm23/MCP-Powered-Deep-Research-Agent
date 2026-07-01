"""
server.py
─────────
FastMCP Server — Deep Research Agent
Registers all research tools and exposes them via the MCP protocol.
Run with: python -m mcp_server.server
"""

import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mcp_server.tools.search_web import search_web as _search_web
from mcp_server.tools.scrape_content import scrape_content as _scrape_content
from mcp_server.tools.summarize import summarize as _summarize

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
log_level = os.getenv("MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mcp_server")

# ── Load Manifest ──────────────────────────────────────────────────────────────
_MANIFEST_PATH = Path(__file__).parent / "manifest.json"
with open(_MANIFEST_PATH) as f:
    _MANIFEST = json.load(f)

logger.info(f"Loaded manifest: {_MANIFEST['name']} v{_MANIFEST['version']}")
logger.info(f"Registered tools: {[t['name'] for t in _MANIFEST['tools']]}")

# ── Create FastMCP App ─────────────────────────────────────────────────────────
mcp = FastMCP(
    name=_MANIFEST["name"],
    version=_MANIFEST["version"],
    instructions=(
        "You are a deep research orchestrator. Use the available tools to "
        "search the web, scrape relevant pages, summarize content, cluster "
        "and rank findings, and generate structured research reports. "
        "Chain tools logically for multi-hop reasoning."
    ),
)


# ── Tool: search_web ──────────────────────────────────────────────────────────
@mcp.tool(
    name="search_web",
    description=(
        "Searches the web for a given query using the Tavily API. "
        "Returns a ranked list of relevant results with titles, URLs, and snippets."
    ),
)
def search_web(query: str, max_results: int = 5, search_depth: str = "basic") -> dict:
    """Search the web and return ranked results."""
    return _search_web(query=query, max_results=max_results, search_depth=search_depth)


# ── Tool: scrape_content ──────────────────────────────────────────────────────
@mcp.tool(
    name="scrape_content",
    description=(
        "Scrapes the full text content from a given URL. "
        "Cleans HTML and returns only readable text, stripping navigation, ads, and boilerplate."
    ),
)
def scrape_content(url: str, max_chars: int = 8000) -> dict:
    """Scrape and clean text content from a URL."""
    return _scrape_content(url=url, max_chars=max_chars)


# ── Tool: summarize ───────────────────────────────────────────────────────────
@mcp.tool(
    name="summarize",
    description=(
        "Summarizes a block of text using an LLM. "
        "Can be focused on a specific aspect (e.g. 'key findings', 'methodology') "
        "to produce targeted research summaries."
    ),
)
def summarize(text: str, focus: str = "", max_length: int = 150) -> dict:
    """Summarize text with an optional focus area."""
    return _summarize(text=text, focus=focus, max_length=max_length)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "localhost")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    logger.info(f"Starting MCP server on {host}:{port} ...")
    mcp.run(transport="stdio")
