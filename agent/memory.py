"""
memory.py
─────────
Shared Memory Module for Zetabot.
Stores intermediate summaries, scraped content, and search results across
reasoning hops so the agent can aggregate context before generating a report.

Context Engineering improvements (ref: Context Engineering Analysis):
  #3  error_log replaces single overwritten `error` string in state → accumulates
  #4  get_relevant_summaries() — query-aware relevance scoring
  #4  get_top_sources() — clean accessor for finalizer
  #6  confidence_scores — per-hop confidence tracking for weighted synthesis
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchMemory:
    """
    In-memory store that accumulates research context across agent hops.
    Passed through the LangGraph state so every node can read/write to it.
    """

    query: str = ""
    search_results: list[dict] = field(default_factory=list)
    scraped_pages: list[dict] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)   # Improvement #6
    tool_call_log: list[dict] = field(default_factory=list)
    error_log: list[str] = field(default_factory=list)             # Improvement #3
    hop_count: int = 0
    final_report: str = ""

    # ── Write helpers ──────────────────────────────────────────────────────────

    def log_tool_call(self, tool_name: str, inputs: dict, output: Any):
        """Record a tool invocation for traceability."""
        self.tool_call_log.append(
            {
                "hop": self.hop_count,
                "tool": tool_name,
                "inputs": inputs,
                "output_preview": str(output)[:300],
            }
        )

    def log_error(self, error: str):
        """Append an error string to the accumulated error log (Improvement #3)."""
        if error and error.strip():
            self.error_log.append(error.strip())

    def add_search_results(self, results: list[dict]):
        """Append new search results, deduplicating by URL."""
        seen_urls = {r["url"] for r in self.search_results}
        for r in results:
            if r["url"] not in seen_urls:
                self.search_results.append(r)
                seen_urls.add(r["url"])

    def add_scraped_page(self, page: dict):
        """Append a scraped page, deduplicating by URL."""
        seen_urls = {p["url"] for p in self.scraped_pages}
        if page["url"] not in seen_urls:
            self.scraped_pages.append(page)

    def add_summary(self, summary: str, confidence: float = 1.0):
        """
        Append a new summary to the memory store.
        Optionally record a confidence score for this hop (Improvement #6).
        """
        if summary and summary.strip():
            self.summaries.append(summary.strip())
            self.confidence_scores.append(max(0.0, min(1.0, confidence)))

    def increment_hop(self):
        self.hop_count += 1

    # ── Read helpers ───────────────────────────────────────────────────────────

    def get_combined_summaries(self) -> str:
        """
        Return all summaries joined (kept for backward compatibility).
        Prefer get_relevant_summaries() for LLM prompts.
        """
        return "\n\n---\n\n".join(self.summaries)

    def get_relevant_summaries(self, query: str, top_n: int = 5) -> list[str]:
        """
        Return the top-N summaries most relevant to the query.
        Ranks by keyword token overlap (Improvement #4).

        Args:
            query: The original research query to score against.
            top_n: Maximum number of summaries to return.

        Returns:
            Ordered list of the most relevant summaries (best-first).
        """
        if not self.summaries:
            return []
        query_words = set(query.lower().split())
        scored = sorted(
            self.summaries,
            key=lambda s: len(query_words & set(s.lower().split())),
            reverse=True,
        )
        return scored[:top_n]

    def get_top_sources(self, top_n: int = 5) -> list[dict]:
        """
        Return the top-N search results by search-engine rank (Improvement #4).
        Results are already ordered by the search tool, so we take the head.
        """
        return self.search_results[:top_n]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "hop_count": self.hop_count,
            "search_result_count": len(self.search_results),
            "scraped_page_count": len(self.scraped_pages),
            "summary_count": len(self.summaries),
            "tool_calls": len(self.tool_call_log),
            "errors": len(self.error_log),
            "has_report": bool(self.final_report),
        }
