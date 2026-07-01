"""
scrape_content.py
─────────────────
MCP Tool: scrape_content
Fetches a URL and returns clean, readable text by stripping HTML boilerplate,
navigation elements, ads, and scripts using BeautifulSoup.
"""

import logging
from typing import Any

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Tags that rarely contain useful content
_NOISE_TAGS = [
    "script", "style", "noscript", "nav", "footer", "header",
    "aside", "form", "button", "iframe", "svg", "figure",
    "advertisement", "ads",
]

# Default HTTP headers to mimic a real browser
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def scrape_content(url: str, max_chars: int = 8000) -> dict[str, Any]:
    """
    Scrape readable text from a webpage URL.

    Args:
        url:       Full URL of the page to scrape.
        max_chars: Maximum characters of text to return (default 8000).

    Returns:
        A dict with keys:
          - url (str)
          - title (str)
          - text (str)      — cleaned body text
          - char_count (int)
          - error (str | None)
    """
    logger.info(f"[scrape_content] Scraping: {url}")

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=_HEADERS) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text

    except httpx.HTTPStatusError as e:
        logger.warning(f"[scrape_content] HTTP error {e.response.status_code} for {url}")
        return {"url": url, "title": "", "text": "", "char_count": 0, "error": str(e)}
    except httpx.RequestError as e:
        logger.warning(f"[scrape_content] Request failed for {url}: {e}")
        return {"url": url, "title": "", "text": "", "char_count": 0, "error": str(e)}

    try:
        soup = BeautifulSoup(html, "lxml")

        # Extract page title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove noisy tags
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        # Try main content containers first
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(id="main-content")
            or soup.find(class_="post-content")
            or soup.find(class_="article-body")
            or soup.body
        )

        raw_text = main_content.get_text(separator="\n") if main_content else soup.get_text(separator="\n")

        # Clean whitespace
        lines = [line.strip() for line in raw_text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)

        # Truncate to max_chars
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "\n\n[Content truncated...]"

        logger.info(f"[scrape_content] Scraped {len(cleaned)} chars from {url}")
        return {
            "url": url,
            "title": title,
            "text": cleaned,
            "char_count": len(cleaned),
            "error": None,
        }

    except Exception as e:
        logger.exception(f"[scrape_content] Parse error for {url}: {e}")
        return {"url": url, "title": "", "text": "", "char_count": 0, "error": str(e)}
