"""
memory.py
─────────
Shared Memory Module for the Deep Research Agent.
Stores intermediate summaries, scraped content, and search results across
reasoning hops so the agent can aggregate context before generating a report.
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
    tool_call_log: list[dict] = field(default_factory=list)
    hop_count: int = 0
    final_report: str = ""

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

    def add_summary(self, summary: str):
        """Append a new summary to the memory store."""
        if summary and summary.strip():
            self.summaries.append(summary.strip())

    def get_combined_summaries(self) -> str:
        """Return all summaries joined for downstream clustering/reporting."""
        return "\n\n---\n\n".join(self.summaries)

    def increment_hop(self):
        self.hop_count += 1

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "hop_count": self.hop_count,
            "search_result_count": len(self.search_results),
            "scraped_page_count": len(self.scraped_pages),
            "summary_count": len(self.summaries),
            "tool_calls": len(self.tool_call_log),
            "has_report": bool(self.final_report),
        }
