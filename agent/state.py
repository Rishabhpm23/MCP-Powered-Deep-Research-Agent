"""
state.py
────────
LangGraph State definition for the Deep Research Agent.
All nodes read from and write to this shared TypedDict.
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from agent.memory import ResearchMemory


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
        error:          Any error message from the last node.
        status:         Human-readable status label for the frontend.
    """

    query: str
    plan: list[str]
    current_step: int
    messages: Annotated[list[Any], add_messages]
    memory: ResearchMemory
    next_action: str   # "search" | "scrape" | "summarize" | "finalize" | "end"
    error: str
    status: str
