"""
summarize.py
────────────
MCP Tool: summarize
Wraps an OpenAI LLM call to produce a focused, concise summary of a text block.
Supports optional focus areas for targeted extraction (e.g. key findings, methodology).
"""

import os
import logging
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def summarize(
    text: str,
    focus: str = "",
    max_length: int = 400,  # Idea 5: raised from 150 → 400 words for richer per-source summaries
) -> dict[str, Any]:
    """
    Summarize a block of text using an OpenAI LLM.

    Args:
        text:       Raw text content to summarize.
        focus:      Optional angle/aspect to focus on (e.g. 'key findings').
        max_length: Target summary length in words (default 150).

    Returns:
        A dict with keys:
          - summary (str)
          - focus (str)
          - word_count (int)
          - model (str)
          - error (str | None)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment.")
        return {
            "summary": "",
            "focus": focus,
            "word_count": 0,
            "model": model,
            "error": "OPENAI_API_KEY is not configured.",
        }

    if not text or not text.strip():
        return {
            "summary": "",
            "focus": focus,
            "word_count": 0,
            "model": model,
            "error": "Input text is empty.",
        }

    # Truncate extremely long inputs to avoid token blowout
    max_input_chars = 12000
    if len(text) > max_input_chars:
        text = text[:max_input_chars] + "\n[Input truncated for summarization]"

    # Build the prompt
    focus_clause = f" Focus specifically on: {focus}." if focus else ""
    system_prompt = (
        "You are an expert research assistant. Your task is to produce a concise, "
        "information-dense summary of the provided text. Preserve key facts, numbers, "
        "names, and conclusions. Avoid filler phrases."
    )
    user_prompt = (
        f"Summarize the following text in approximately {max_length} words.{focus_clause}\n\n"
        f"TEXT:\n{text}"
    )

    try:
        client = OpenAI(api_key=api_key)
        logger.info(f"[summarize] Calling {model} | focus='{focus}' | target_words={max_length}")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=max_length * 2,  # generous token budget
        )

        summary = response.choices[0].message.content.strip()
        word_count = len(summary.split())

        logger.info(f"[summarize] Generated summary: {word_count} words.")
        return {
            "summary": summary,
            "focus": focus,
            "word_count": word_count,
            "model": model,
            "error": None,
        }

    except Exception as e:
        logger.exception(f"[summarize] OpenAI call failed: {e}")
        return {
            "summary": "",
            "focus": focus,
            "word_count": 0,
            "model": model,
            "error": str(e),
        }
