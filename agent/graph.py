"""
graph.py
────────
LangGraph State Graph — wires all agent nodes together into a
directed research workflow with conditional routing.

Graph flow:
  START → planner → tool_caller → [loop or memory_aggregator] → finalizer → END
"""

import logging
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import (
    planner_node,
    tool_caller_node,
    memory_aggregator_node,
    finalizer_node,
)

logger = logging.getLogger(__name__)


def _route_after_tool(state: AgentState) -> str:
    """
    Conditional router after tool_caller_node.
    - If there are more steps → loop back to tool_caller
    - If all steps done → go to memory_aggregator
    """
    action = state.get("next_action", "execute")
    if action == "execute":
        return "tool_caller"
    return "memory_aggregator"


def _route_after_aggregator(state: AgentState) -> str:
    """Always route to finalizer after aggregation."""
    return "finalizer"


def build_research_graph() -> StateGraph:
    """
    Construct and compile the full research agent graph.
    Returns a compiled LangGraph runnable.
    """
    builder = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────────
    builder.add_node("planner", planner_node)
    builder.add_node("tool_caller", tool_caller_node)
    builder.add_node("memory_aggregator", memory_aggregator_node)
    builder.add_node("finalizer", finalizer_node)

    # ── Add edges ──────────────────────────────────────────────────────────────
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "tool_caller")

    # Conditional loop: keep calling tools until plan is exhausted
    builder.add_conditional_edges(
        "tool_caller",
        _route_after_tool,
        {"tool_caller": "tool_caller", "memory_aggregator": "memory_aggregator"},
    )

    builder.add_edge("memory_aggregator", "finalizer")
    builder.add_edge("finalizer", END)

    graph = builder.compile()
    logger.info("[Graph] Research agent graph compiled successfully.")
    return graph


# ── Module-level compiled graph (singleton) ────────────────────────────────────
research_graph = build_research_graph()
