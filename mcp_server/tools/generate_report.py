"""
generate_report.py
──────────────────
MCP Tool: generate_report
Compiles ranked research findings into a structured output document.
Supports three formats:
  - markdown_brief:    Concise executive-style Markdown report
  - comparison_table:  Side-by-side Markdown table of key attributes
  - insight_report:    Detailed narrative with actionable insights
"""

import os
import logging
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def generate_report(
    findings: list[str],
    query: str,
    format: str = "markdown_brief",
) -> dict[str, Any]:
    """
    Generate a structured research report from a list of findings.

    Args:
        findings: List of ranked research findings or clustered summaries.
        query:    The original research query (used as title/context).
        format:   Output format — 'markdown_brief' | 'comparison_table' | 'insight_report'.

    Returns:
        A dict with keys:
          - report (str)       — the formatted report content
          - format (str)
          - word_count (int)
          - error (str | None)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        return {
            "report": "",
            "format": format,
            "word_count": 0,
            "error": "OPENAI_API_KEY is not configured.",
        }

    if not findings:
        return {
            "report": "",
            "format": format,
            "word_count": 0,
            "error": "No findings provided to generate report.",
        }

    combined_findings = "\n\n---\n\n".join(
        f"**Finding {i+1}:**\n{f}" for i, f in enumerate(findings)
    )

    # ── Build format-specific prompt ──────────────────────────────────────────
    format_instructions = _get_format_instructions(format, query)

    system = (
        "You are an expert research analyst and technical writer. "
        "Produce professional, accurate, well-structured research documents. "
        "Use precise language, cite specifics, and be comprehensive yet concise."
    )

    user_prompt = (
        f"Research Query: {query}\n\n"
        f"Research Findings:\n{combined_findings}\n\n"
        f"{format_instructions}"
    )

    try:
        client = OpenAI(api_key=api_key)
        logger.info(f"[generate_report] Generating '{format}' report for: '{query}'")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=5000,  # Idea 4: raised from 2000 → 5000 to remove report length ceiling
        )

        report = response.choices[0].message.content.strip()
        word_count = len(report.split())
        logger.info(f"[generate_report] Report generated: {word_count} words ({format})")

        return {
            "report": report,
            "format": format,
            "word_count": word_count,
            "error": None,
        }

    except Exception as e:
        logger.exception(f"[generate_report] Error: {e}")
        return {
            "report": "",
            "format": format,
            "word_count": 0,
            "error": str(e),
        }


def _get_format_instructions(format: str, query: str) -> str:
    """Return format-specific LLM instructions."""

    if format == "markdown_brief":
        return f"""Generate a concise Markdown research brief with these sections:
## {query}
### Executive Summary
(2-3 sentence overview)
### Key Findings
- (bullet points with the most important facts)
### Analysis
(2-3 paragraphs of synthesis)
### Conclusion
(1-2 sentences with the main takeaway)
### Sources Referenced
(list any URLs or sources mentioned in findings)"""

    elif format == "comparison_table":
        return f"""Generate a Markdown comparison table analyzing the research findings.
Structure:
1. Brief introduction (1-2 sentences about: {query})
2. A Markdown table comparing key attributes across findings/sources
   Columns should capture: Aspect | Details | Significance | Source/Evidence
3. Summary paragraph highlighting the most important comparisons"""

    elif format == "insight_report":
        return f"""Generate a detailed insight report with actionable intelligence about: {query}
Structure:
## Research Insight Report: {query}
### Background & Context
### Core Insights (numbered, with supporting evidence)
### Emerging Trends
### Challenges & Limitations
### Recommendations & Next Steps
### Conclusion
Use subheadings, bullet points, and **bold** for emphasis."""

    else:
        return f"Generate a clear, well-organized research summary about: {query}"


def compare_sources(sources: list[dict], query: str = "") -> dict[str, Any]:
    """
    Convenience function: takes a list of source dicts (with title, url, content)
    and generates a comparison table report.

    Args:
        sources: List of dicts with keys: title, url, content/summary
        query:   Context query for framing the comparison

    Returns:
        Same structure as generate_report output.
    """
    if not sources:
        return {"report": "", "format": "comparison_table", "word_count": 0, "error": "No sources provided."}

    findings = [
        f"Source: {s.get('title', 'Unknown')} ({s.get('url', 'N/A')})\n{s.get('content') or s.get('summary', '')}"
        for s in sources
    ]
    return generate_report(findings=findings, query=query or "Source Comparison", format="comparison_table")
