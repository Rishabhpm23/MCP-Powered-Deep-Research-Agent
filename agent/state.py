"""
state.py
────────
LangGraph State definition for Zetabot.
All nodes read from and write to this shared TypedDict.

Context Engineering improvement (ref: Context Engineering Analysis):
  #3  `errors` is now a list[str] that accumulates across hops,
      replacing the single overwritten `error: str` field.
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from agent.memory import ResearchMemory


def _accumulate_errors(left: list[str], right: list[str]) -> list[str]:
    """Reducer: append new errors to the existing list (never overwrite)."""
    return left + [e for e in right if e]


class AgentState(TypedDict):
    """
    The shared state passed between all LangGraph nodes.

    Fields:
        query:          The original user research query.
        plan:           Ordered list of research steps (from planner node).
        current_step:   Index of the step currently being executed.
        messages:       LangChain messages (accumulate via add_messages).
        memory:         ResearchMemory object holding all intermediate outputs.
        next_action:    Routing signal — which node to visit next.
        errors:         Accumulated list of error strings across all hops.
                        Never overwritten — new errors are appended.
        status:         Human-readable status label for the frontend.
    """

    query: str
    plan: list[str]
    current_step: int
    messages: Annotated[list[Any], add_messages]
    memory: ResearchMemory
    next_action: str   # "search" | "scrape" | "summarize" | "finalize" | "end"
    errors: Annotated[list[str], _accumulate_errors]   # was: error: str
    status: str
