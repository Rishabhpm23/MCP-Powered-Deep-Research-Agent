"""
nodes.py
────────
LangGraph Node implementations for the Deep Research Agent.

Nodes:
  - planner_node         : Uses LLM to break query into research steps
  - tool_caller_node     : Executes the appropriate MCP tool for current step
  - memory_aggregator_node: Combines intermediate outputs in shared memory
  - finalizer_node       : Synthesizes final research report

Each node receives AgentState and returns a dict of state updates.
"""

import os
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

from agent.state import AgentState
from agent.memory import ResearchMemory
from agent.mcp_client import MCPClient

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


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: Planner
# ─────────────────────────────────────────────────────────────────────────────
def planner_node(state: AgentState) -> dict:
    """
    Uses the LLM to decompose the research query into a concrete, ordered
    list of research steps. Each step maps naturally to an MCP tool.
    """
    logger.info(f"[Planner] Planning research for: '{state['query']}'")

    system = SystemMessage(content="""You are a research planning assistant. 
Given a research query, output a JSON array of research steps. 
Each step is a short instruction string. Steps should be ordered logically.
Available tools: search_web, scrape_content, summarize.
Example output:
["Search for recent papers on topic X", "Scrape the top 3 results", "Summarize each page focusing on key findings", "Summarize all results together"]
Output ONLY valid JSON — no markdown, no explanation.""")

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
        }
    except Exception as e:
        logger.error(f"[Planner] Failed to parse plan: {e}")
        # Fallback plan
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
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: Tool Caller
# ─────────────────────────────────────────────────────────────────────────────
def tool_caller_node(state: AgentState) -> dict:
    """
    Interprets the current plan step and decides which MCP tool to call,
    then executes it and stores results in shared memory.
    """
    memory: ResearchMemory = state["memory"]
    plan = state["plan"]
    step_idx = state["current_step"]

    if step_idx >= len(plan):
        return {"next_action": "finalize", "status": "All steps complete"}

    current_step = plan[step_idx]
    logger.info(f"[ToolCaller] Step {step_idx + 1}/{len(plan)}: {current_step}")
    memory.increment_hop()

    # ── LLM decides which tool to use ─────────────────────────────────────────
    system = SystemMessage(content="""You are a tool-calling assistant for a research agent.
Given a research step description and context, decide which tool to call and with what arguments.
Available tools and their arguments:
- search_web(query: str, max_results: int=5)
- scrape_content(url: str, max_chars: int=8000)  
- summarize(text: str, focus: str="", max_length: int=150)

Context available:
- Recent search URLs (for scraping)
- Recent scraped texts (for summarizing)

Output ONLY valid JSON like:
{"tool": "search_web", "args": {"query": "...", "max_results": 5}}
No markdown, no explanation.""")

    # Build context string
    context_parts = []
    if memory.search_results:
        recent_urls = [r["url"] for r in memory.search_results[-5:]]
        context_parts.append(f"Recent search URLs: {recent_urls}")
    if memory.scraped_pages:
        recent_text = memory.scraped_pages[-1]["text"][:500]
        context_parts.append(f"Recent scraped text (first 500 chars): {recent_text}")

    context_str = "\n".join(context_parts) if context_parts else "No prior context yet."
    human = HumanMessage(
        content=f"Step to execute: {current_step}\n\nContext:\n{context_str}"
    )

    try:
        response = _get_llm().invoke([system, human])
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

    # ── Execute the MCP tool ───────────────────────────────────────────────────
    logger.info(f"[ToolCaller] Calling MCP tool: {tool_name}({tool_args})")
    result = _MCP.call(tool_name, **tool_args)
    memory.log_tool_call(tool_name, tool_args, result)

    # ── Store results in memory ────────────────────────────────────────────────
    if tool_name == "search_web" and not result.get("error"):
        memory.add_search_results(result.get("results", []))
        status = f"Found {result['total_results']} results for '{tool_args.get('query', '')}'"

    elif tool_name == "scrape_content" and not result.get("error"):
        memory.add_scraped_page(result)
        status = f"Scraped {result['char_count']} chars from {tool_args.get('url', '')}"

    elif tool_name == "summarize" and not result.get("error"):
        memory.add_summary(result.get("summary", ""))
        status = f"Summary generated: {result['word_count']} words"

    else:
        status = f"Tool {tool_name} error: {result.get('error', 'unknown')}"

    next_step = step_idx + 1
    next_action = "execute" if next_step < len(plan) else "finalize"

    return {
        "current_step": next_step,
        "memory": memory,
        "next_action": next_action,
        "status": status,
        "messages": [response] if "response" in dir() else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: Memory Aggregator
# ─────────────────────────────────────────────────────────────────────────────
def memory_aggregator_node(state: AgentState) -> dict:
    """
    Combines all scraped pages into a single consolidated summary if needed.
    Runs between the last tool call step and finalization.
    """
    memory: ResearchMemory = state["memory"]
    logger.info(f"[MemoryAgg] Aggregating {len(memory.scraped_pages)} pages, {len(memory.summaries)} summaries.")

    # If we have scraped pages but no summaries yet, auto-summarize them
    if memory.scraped_pages and not memory.summaries:
        for page in memory.scraped_pages[:3]:  # limit to 3 pages
            if page["text"]:
                result = _MCP.call(
                    "summarize",
                    text=page["text"],
                    focus="key findings and important facts",
                    max_length=120,
                )
                if not result.get("error"):
                    memory.add_summary(result["summary"])
                    memory.log_tool_call("summarize", {"url": page["url"]}, result)

    return {
        "memory": memory,
        "next_action": "finalize",
        "status": f"Memory aggregated: {len(memory.summaries)} summaries ready",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: Finalizer
# ─────────────────────────────────────────────────────────────────────────────
def finalizer_node(state: AgentState) -> dict:
    """
    Synthesizes all memory into a well-structured Markdown research report
    using the LLM. This is the terminal node that produces the final output.
    """
    memory: ResearchMemory = state["memory"]
    logger.info(f"[Finalizer] Generating report for: '{state['query']}'")

    combined_summaries = memory.get_combined_summaries()
    search_snippet = ""
    if memory.search_results:
        top_results = memory.search_results[:5]
        search_snippet = "\n".join(
            f"- [{r['title']}]({r['url']}): {r['content'][:200]}" for r in top_results
        )

    system = SystemMessage(content="""You are an expert research analyst. 
Generate a comprehensive, well-structured Markdown research report.
Include:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis (paragraphs)
4. Sources (numbered list of URLs)
5. Conclusion

Use proper Markdown formatting with headers (##), bold, and bullet points.
Be factual, concise, and cite sources inline where possible.""")

    human = HumanMessage(content=f"""Research Query: {state['query']}

Summaries gathered:
{combined_summaries if combined_summaries else 'No summaries available.'}

Top sources found:
{search_snippet if search_snippet else 'No sources available.'}

Generate the full research report now.""")

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
        }
    except Exception as e:
        logger.error(f"[Finalizer] Report generation failed: {e}")
        # Fallback: return raw summaries
        fallback_report = f"# Research Report: {state['query']}\n\n{combined_summaries}"
        memory.final_report = fallback_report
        return {
            "memory": memory,
            "next_action": "end",
            "status": "Research complete (fallback)",
            "error": str(e),
        }
