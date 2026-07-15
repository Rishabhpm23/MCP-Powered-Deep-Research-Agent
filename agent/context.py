"""
context.py
──────────
ContextBuilder — Context Engineering Utility for Zetabot.

Replaces the ad-hoc, per-node context assembly with a single, structured
utility that provides:
  - Token-budget-aware context windows (no unbounded dumps to the LLM)
  - Relevance-ranked selection (keyword overlap with the original query)
  - Rich tool-call history for the tool caller node
  - Curated source snippets for the finalizer node

Improvements implemented (ref: Context Engineering Analysis):
  #1  Structured ContextBuilder replacing raw slices in nodes.py
  #2  Dynamic system prompts via build_*_system_prompt() helpers
  #4  Query-aware relevance scoring (supplements memory.py methods)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.memory import ResearchMemory

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for GPT-class models
_CHARS_PER_TOKEN = 4


class ContextBuilder:
    """
    Assembles structured, token-budget-aware context blocks for each LangGraph node.

    Args:
        query:      The original user research query.
        max_hops:   The maximum number of hops configured for this session.
        max_tokens: Soft token budget for context blocks (default 3 000).
    """

    def __init__(self, query: str, max_hops: int = 5, max_tokens: int = 3000):
        self.query = query
        self.max_hops = max_hops
        self.max_tokens = max_tokens
        self._char_budget = max_tokens * _CHARS_PER_TOKEN

    # ─────────────────────────────────────────────────────────────────────────
    # System prompt builders  (Improvement #2 — dynamic system prompts)
    # ─────────────────────────────────────────────────────────────────────────

    def build_planner_system_prompt(self, available_tools: list[str]) -> str:
        """
        Dynamic system prompt for planner_node.
        Injects MAX_HOPS and the live tool list so the LLM plans within bounds.
        """
        tool_list = ", ".join(available_tools)
        return (
            f"You are a research planning assistant.\n"
            f"Session constraints:\n"
            f"  - Maximum research hops available: {self.max_hops}\n"
            f"  - Available tools: {tool_list}\n"
            f"  - Goal: produce a focused, comprehensive answer to the query\n\n"
            f"Given the research query, output a JSON array of AT MOST {self.max_hops} steps.\n"
            f"Each step must be a short, actionable instruction string that maps to one of the "
            f"available tools ({tool_list}).\n"
            f"Order steps logically: gather → scrape → summarize → synthesize.\n"
            f"Output ONLY valid JSON — no markdown, no explanation.\n"
            f"Example: [\"Search for recent papers on X\", \"Scrape the top 3 results\", "
            f"\"Summarize each page\", \"Summarize all results together\"]"
        )

    def build_tool_caller_system_prompt(self, memory: ResearchMemory) -> str:
        """
        Dynamic system prompt for tool_caller_node.
        Injects hop budget, current progress, and a strategic nudge based on
        how many hops remain.
        """
        hops_remaining = self.max_hops - memory.hop_count
        progress = (
            f"  - Summaries generated: {len(memory.summaries)}\n"
            f"  - Sources found: {len(memory.search_results)}\n"
            f"  - Pages scraped: {len(memory.scraped_pages)}\n"
            f"  - Hops remaining: {hops_remaining}/{self.max_hops}"
        )

        # Strategic nudge based on remaining budget
        if hops_remaining <= 2 and memory.search_results:
            strategy = (
                "PRIORITY: You are running low on hops. "
                "PREFER summarize or scrape over new searches. "
                "Consolidate what you already have."
            )
        elif not memory.search_results:
            strategy = "No searches done yet — start with search_web."
        else:
            strategy = "Continue gathering diverse, high-quality sources."

        return (
            f"You are a tool-calling assistant for a research agent.\n"
            f"Current session progress:\n{progress}\n\n"
            f"Strategic guidance: {strategy}\n\n"
            f"Available tools and their arguments:\n"
            f"  - search_web(query: str, max_results: int=10, search_depth: str='advanced')\n"  # Idea 2: advanced depth + 10 results
            f"  - scrape_content(url: str, max_chars: int=8000)\n"
            f"  - summarize(text: str, focus: str='', max_length: int=400)\n\n"  # Idea 5: 400-word summaries
            f"IMPORTANT: For search_web, always use search_depth='advanced' and max_results=10 unless hops are critically low.\n\n"
            f"Given the step description and context, decide which tool to call.\n"
            f"Output ONLY valid JSON:\n"
            f'  {{"tool": "<tool_name>", "args": {{...}}}}\n'
            f"No markdown, no explanation."
        )

    def build_finalizer_system_prompt(self, memory: ResearchMemory) -> str:
        """
        Dynamic system prompt for finalizer_node.
        Adapts depth/persona based on how much data was actually gathered.
        Idea 3: Expanded with 9 mandatory sections and explicit word-count targets
        to force a comprehensive 1500–2000+ word report.
        """
        data_richness = len(memory.summaries) + len(memory.search_results)
        if data_richness >= 8:
            depth = (
                "You have rich, multi-source data available. Be comprehensive, detailed, and "
                "thorough. Write as an expert analyst publishing a professional research document."
            )
        elif data_richness >= 3:
            depth = (
                "You have moderate data available. Be thorough and specific. "
                "Expand each section fully with all available evidence."
            )
        else:
            depth = (
                "Data is limited. Be honest about gaps, cite what you have, "
                "and do not fabricate details. Expand on what is available."
            )

        return (
            f"You are an expert research analyst and technical writer. {depth}\n\n"
            f"Generate a comprehensive, well-structured Markdown research report with ALL of the "
            f"following 9 sections. Do NOT skip any section. Each section must meet its minimum "
            f"word count — this is a professional research document, not a summary.\n\n"
            f"Required sections (in order):\n"
            f"  ## 1. Executive Summary (minimum 150 words)\n"
            f"     A thorough overview of what was researched, why it matters, and the key conclusion.\n\n"
            f"  ## 2. Background & Context (minimum 150 words)\n"
            f"     Why is this comparison/topic important? What is the broader landscape?\n\n"
            f"  ## 3. Key Findings (minimum 10 bullet points, each with 1–2 sentences of evidence)\n"
            f"     Concrete, specific findings — numbers, benchmarks, quotes — not vague statements.\n\n"
            f"  ## 4. Capability Deep-Dive (minimum 300 words)\n"
            f"     Separate subsections (###) for each major dimension relevant to the query\n"
            f"     (e.g., reasoning, coding, speed, cost, safety, multimodal, context window, API).\n\n"
            f"  ## 5. Comparative Analysis (minimum 200 words)\n"
            f"     Head-to-head comparison with a Markdown table where data supports it.\n"
            f"     Columns: Dimension | Option A | Option B | Winner/Notes\n\n"
            f"  ## 6. Use Case Recommendations (minimum 150 words)\n"
            f"     Specific scenarios and which option is best suited for each, with reasoning.\n\n"
            f"  ## 7. Limitations & Research Gaps (minimum 100 words)\n"
            f"     Honest gaps in the research, caveats, or areas needing more investigation.\n\n"
            f"  ## 8. Sources (all sources, numbered)\n"
            f"     Full numbered list with URL and a one-line description of each source.\n\n"
            f"  ## 9. Conclusion (minimum 150 words)\n"
            f"     Synthesize all findings into a decisive, actionable conclusion.\n\n"
            f"Formatting rules:\n"
            f"  - Use proper Markdown: ##, ###, **bold**, bullet points, tables\n"
            f"  - Cite sources inline as (Source: [Title](URL))\n"
            f"  - Be factual — do not fabricate data\n"
            f"  - Total report must be at minimum 1200 words"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Context block builders  (Improvement #1 — ContextBuilder)
    # ─────────────────────────────────────────────────────────────────────────

    def build_tool_caller_context(self, memory: ResearchMemory) -> str:
        """
        Structured context block for tool_caller_node.

        Replaces the old naive 500-char slice. Now includes:
          - Original query (anchor)
          - Full tool-call history (last 5 hops, 150-char previews)
          - All already-searched URLs (to avoid re-querying)
          - Latest summary preview (what we know so far)
          - Errors encountered (so the LLM can avoid repeating failures)
        """
        parts: list[str] = [f"Research Goal: {self.query}"]

        # Tool call history — what has been tried
        if memory.tool_call_log:
            history_lines = [
                f"  Hop {c['hop']}: {c['tool']}({c['inputs']}) → {c['output_preview'][:150]}"
                for c in memory.tool_call_log[-5:]
            ]
            parts.append("Recent Tool History:\n" + "\n".join(history_lines))

        # All found URLs — prevent the LLM from re-querying the same sources
        if memory.search_results:
            all_urls = [r["url"] for r in memory.search_results]
            parts.append(
                f"Already-found URLs ({len(all_urls)} total — do NOT re-scrape these):\n"
                + "\n".join(f"  - {u}" for u in all_urls[-10:])
            )

        # Summaries built so far — what we already know
        if memory.summaries:
            parts.append(
                f"Summaries built so far ({len(memory.summaries)} total).\n"
                f"Latest summary preview:\n  {memory.summaries[-1][:300]}"
            )

        # Error history — don't repeat mistakes
        if memory.error_log:
            parts.append(
                "Previous errors (avoid repeating these):\n"
                + "\n".join(f"  - {e}" for e in memory.error_log[-3:])
            )

        return "\n\n".join(parts)

    def build_finalizer_context(self, memory: ResearchMemory) -> str:
        """
        Token-capped, relevance-ranked context for finalizer_node.

        Replaces the old get_combined_summaries() unbounded dump. Now:
          - Scores each summary by keyword overlap with the query
          - Selects top summaries within the char budget
          - Appends a curated sources block
        """
        # Score summaries by query-keyword overlap
        query_tokens = set(self.query.lower().split())
        scored: list[tuple[float, str]] = []
        for summary in memory.summaries:
            summary_tokens = set(summary.lower().split())
            overlap = len(query_tokens & summary_tokens)
            score = overlap / max(len(query_tokens), 1)
            scored.append((score, summary))

        # Sort best-first, then fill char budget
        scored.sort(key=lambda x: x[0], reverse=True)
        selected_summaries: list[str] = []
        chars_used = 0
        for _score, s in scored:
            if chars_used + len(s) > self._char_budget:
                logger.debug(
                    "[ContextBuilder] Char budget reached — truncating summaries at "
                    "%d/%d", chars_used, self._char_budget
                )
                break
            selected_summaries.append(s)
            chars_used += len(s)

        summaries_block = (
            "\n\n---\n\n".join(selected_summaries)
            if selected_summaries
            else "No summaries available."
        )

        # Curated sources block (top-5 by search-engine rank, via memory method)
        top_sources = memory.get_top_sources(top_n=5)
        sources_block = (
            "\n".join(
                f"- [{r['title']}]({r['url']}): {r.get('content', '')[:200]}"
                for r in top_sources
            )
            if top_sources
            else "No sources available."
        )

        logger.info(
            "[ContextBuilder] Finalizer context: %d/%d summaries selected (%d chars), "
            "%d sources",
            len(selected_summaries), len(memory.summaries), chars_used, len(top_sources)
        )

        return summaries_block, sources_block
