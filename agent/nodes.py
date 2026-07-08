"""
nodes.py
────────
LangGraph Node implementations for the Deep Research Agent.

Nodes:
  - planner_node          : Uses LLM to break query into research steps
  - tool_caller_node      : Executes the appropriate MCP tool for current step
  - memory_aggregator_node: Combines intermediate outputs in shared memory
  - finalizer_node        : Synthesizes final research report

Each node receives AgentState and returns a dict of state updates.

Context Engineering improvements applied (ref: Context Engineering Analysis):
  #1  tool_caller_node uses ContextBuilder.build_tool_caller_context()
      instead of a naive 500-char slice of the last scraped page.
  #2  All nodes use dynamic system prompts (hop budget, tools, data richness).
  #3  Errors are appended to state["errors"] list, never overwritten.
  #4  finalizer_node uses ContextBuilder.build_finalizer_context() which
      token-caps and relevance-ranks summaries via keyword overlap scoring.
  #5  tool_caller_node injects state["messages"][-4:] for conversational memory.
  #6  add_summary() records per-hop confidence scores.
"""

import os
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from agent.state import AgentState
from agent.memory import ResearchMemory
from agent.mcp_client import MCPClient
from agent.context import ContextBuilder

load_dotenv()
logger = logging.getLogger(__name__)

# ── Lazy LLM getter (avoids import-time failure when .env not set) ────────────
_llm_instance = None


def _get_llm() -> ChatOpenAI:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm_instance


# ── Shared MCP client ──────────────────────────────────────────────────────────
_MCP = MCPClient()

MAX_HOPS = int(os.getenv("MAX_HOPS", "5"))


def _make_context_builder(state: AgentState) -> ContextBuilder:
    """Construct a ContextBuilder bound to this session's query and hop limit."""
    return ContextBuilder(
        query=state["query"],
        max_hops=MAX_HOPS,
        max_tokens=3000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: Planner
# ─────────────────────────────────────────────────────────────────────────────
def planner_node(state: AgentState) -> dict:
    """
    Uses the LLM to decompose the research query into a concrete, ordered
    list of research steps. Each step maps naturally to an MCP tool.

    Improvement #2: System prompt is dynamic — injects MAX_HOPS and the live
    list of available tools so the planner stays within session constraints.
    """
    logger.info(f"[Planner] Planning research for: '{state['query']}'")

    ctx = _make_context_builder(state)

    # Dynamic system prompt — knows hop budget and available tools (#2)
    system = SystemMessage(
        content=ctx.build_planner_system_prompt(available_tools=_MCP.list_tools())
    )
    human = HumanMessage(content=f"Research query: {state['query']}")

    try:
        response = _get_llm().invoke([system, human])
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        plan = json.loads(raw.strip())
        if not isinstance(plan, list):
            raise ValueError("Plan must be a list")
        # Cap to MAX_HOPS steps
        plan = plan[:MAX_HOPS]
        logger.info(f"[Planner] Generated {len(plan)} steps: {plan}")
        return {
            "plan": plan,
            "current_step": 0,
            "next_action": "execute",
            "status": f"Plan ready: {len(plan)} steps",
            "messages": [response],
            "errors": [],
        }
    except Exception as e:
        logger.error(f"[Planner] Failed to parse plan: {e}")
        fallback = [
            f"Search the web for: {state['query']}",
            "Scrape the top result",
            "Summarize the scraped content",
        ]
        return {
            "plan": fallback,
            "current_step": 0,
            "next_action": "execute",
            "status": "Using fallback plan",
            "errors": [f"[planner] {e}"],   # Improvement #3 — append, not overwrite
        }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: Tool Caller
# ─────────────────────────────────────────────────────────────────────────────
def tool_caller_node(state: AgentState) -> dict:
    """
    Interprets the current plan step and decides which MCP tool to call,
    then executes it and stores results in shared memory.

    Improvements applied:
      #1  Uses ContextBuilder.build_tool_caller_context() — full history,
          all found URLs, latest summary, error log; not a 500-char slice.
      #2  Dynamic system prompt — injects hop budget and strategic nudge.
      #5  Injects state["messages"][-4:] as conversational memory.
      #6  Confidence score derived from tool success for add_summary().
    """
    memory: ResearchMemory = state["memory"]
    plan = state["plan"]
    step_idx = state["current_step"]

    if step_idx >= len(plan):
        return {"next_action": "finalize", "status": "All steps complete", "errors": []}

    current_step = plan[step_idx]
    logger.info(f"[ToolCaller] Step {step_idx + 1}/{len(plan)}: {current_step}")
    memory.increment_hop()

    ctx = _make_context_builder(state)

    # Dynamic system prompt — hop budget + strategic nudge (#2)
    system = SystemMessage(
        content=ctx.build_tool_caller_system_prompt(memory)
    )

    # Rich context block (#1) — history, all URLs, summaries, error log
    context_str = ctx.build_tool_caller_context(memory)

    human = HumanMessage(
        content=f"Step to execute: {current_step}\n\nContext:\n{context_str}"
    )

    # Conversational memory — last 4 messages for continuity (#5)
    recent_messages = state.get("messages", [])[-4:]

    try:
        response = _get_llm().invoke([system] + recent_messages + [human])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw.strip())
        tool_name = decision["tool"]
        tool_args = decision.get("args", {})
    except Exception as e:
        logger.error(f"[ToolCaller] LLM decision failed: {e}. Using search fallback.")
        tool_name = "search_web"
        tool_args = {"query": state["query"], "max_results": 3}
        # Log the decision error (#3)
        memory.log_error(f"[tool_caller decision] {e}")

    # ── Execute the MCP tool ───────────────────────────────────────────────────
    logger.info(f"[ToolCaller] Calling MCP tool: {tool_name}({tool_args})")
    result = _MCP.call(tool_name, **tool_args)
    memory.log_tool_call(tool_name, tool_args, result)

    # ── Store results in memory & compute confidence (#6) ─────────────────────
    tool_error = result.get("error")
    new_errors: list[str] = []

    if tool_name == "search_web" and not tool_error:
        memory.add_search_results(result.get("results", []))
        status = f"Found {result['total_results']} results for '{tool_args.get('query', '')}'"

    elif tool_name == "scrape_content" and not tool_error:
        memory.add_scraped_page(result)
        status = f"Scraped {result['char_count']} chars from {tool_args.get('url', '')}"

    elif tool_name == "summarize" and not tool_error:
        # Confidence = word_count normalised to max 200 words (#6)
        word_count = result.get("word_count", 0)
        confidence = min(word_count / 200.0, 1.0)
        memory.add_summary(result.get("summary", ""), confidence=confidence)
        status = f"Summary generated: {word_count} words (confidence={confidence:.2f})"

    else:
        err_msg = f"[tool:{tool_name}] {tool_error or 'unknown error'}"
        status = f"Tool {tool_name} error: {tool_error or 'unknown'}"
        memory.log_error(err_msg)         # accumulate in memory (#3)
        new_errors = [err_msg]            # propagate to state (#3)

    next_step = step_idx + 1
    next_action = "execute" if next_step < len(plan) else "finalize"

    return {
        "current_step": next_step,
        "memory": memory,
        "next_action": next_action,
        "status": status,
        "messages": [response] if "response" in dir() else [],
        "errors": new_errors,   # Improvement #3 — reducer appends, never overwrites
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: Memory Aggregator
# ─────────────────────────────────────────────────────────────────────────────
def memory_aggregator_node(state: AgentState) -> dict:
    """
    Combines all scraped pages into a single consolidated summary if needed.
    Runs between the last tool call step and finalization.

    Improvement #6: confidence score passed to add_summary() based on
    word count of the generated summary.
    """
    memory: ResearchMemory = state["memory"]
    logger.info(
        f"[MemoryAgg] Aggregating {len(memory.scraped_pages)} pages, "
        f"{len(memory.summaries)} summaries."
    )

    new_errors: list[str] = []

    # If we have scraped pages but no summaries yet, auto-summarize them
    if memory.scraped_pages and not memory.summaries:
        for page in memory.scraped_pages[:3]:   # limit to 3 pages
            if page["text"]:
                result = _MCP.call(
                    "summarize",
                    text=page["text"],
                    focus="key findings and important facts",
                    max_length=120,
                )
                if not result.get("error"):
                    word_count = result.get("word_count", 0)
                    confidence = min(word_count / 200.0, 1.0)       # Improvement #6
                    memory.add_summary(result["summary"], confidence=confidence)
                    memory.log_tool_call("summarize", {"url": page["url"]}, result)
                else:
                    err_msg = f"[aggregator summarize:{page['url']}] {result['error']}"
                    memory.log_error(err_msg)
                    new_errors.append(err_msg)

    return {
        "memory": memory,
        "next_action": "finalize",
        "status": f"Memory aggregated: {len(memory.summaries)} summaries ready",
        "errors": new_errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: Finalizer
# ─────────────────────────────────────────────────────────────────────────────
def finalizer_node(state: AgentState) -> dict:
    """
    Synthesizes all memory into a well-structured Markdown research report
    using the LLM. This is the terminal node that produces the final output.

    Improvements applied:
      #2  Dynamic system prompt — adapts depth/persona to data richness.
      #4  Uses ContextBuilder.build_finalizer_context() which token-caps
          and relevance-ranks summaries instead of dumping all of them.
      #3  Error propagation uses errors list.
    """
    memory: ResearchMemory = state["memory"]
    logger.info(f"[Finalizer] Generating report for: '{state['query']}'")

    ctx = _make_context_builder(state)

    # Dynamic system prompt — adapts to how much data was gathered (#2)
    system = SystemMessage(
        content=ctx.build_finalizer_system_prompt(memory)
    )

    # Token-capped, relevance-ranked context — not an unbounded dump (#4)
    summaries_block, sources_block = ctx.build_finalizer_context(memory)

    human = HumanMessage(
        content=(
            f"Research Query: {state['query']}\n\n"
            f"Summaries gathered (relevance-ranked, token-budgeted):\n"
            f"{summaries_block}\n\n"
            f"Top sources found:\n"
            f"{sources_block}\n\n"
            f"Generate the full research report now."
        )
    )

    try:
        response = _get_llm().invoke([system, human])
        report = response.content.strip()
        memory.final_report = report
        logger.info(f"[Finalizer] Report generated: {len(report)} chars")
        return {
            "memory": memory,
            "next_action": "end",
            "status": "Research complete ✓",
            "messages": [response],
            "errors": [],
        }
    except Exception as e:
        logger.error(f"[Finalizer] Report generation failed: {e}")
        fallback_report = (
            f"# Research Report: {state['query']}\n\n"
            f"{memory.get_combined_summaries()}"
        )
        memory.final_report = fallback_report
        err_msg = f"[finalizer] {e}"
        memory.log_error(err_msg)
        return {
            "memory": memory,
            "next_action": "end",
            "status": "Research complete (fallback)",
            "errors": [err_msg],    # Improvement #3
        }
