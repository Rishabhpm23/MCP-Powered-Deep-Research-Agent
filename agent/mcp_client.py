"""
mcp_client.py
─────────────
MCP Client — calls MCP server tools directly (in-process for Phase 2).
Uses direct Python imports rather than JSON-RPC transport so the agent
can run the full stack in a single process during development.

In Phase 5, this will be swapped for a true MCP transport client.
"""

import logging
from typing import Any

from mcp_server.tools.search_web import search_web
from mcp_server.tools.scrape_content import scrape_content
from mcp_server.tools.summarize import summarize

logger = logging.getLogger(__name__)

# Registry maps tool names → callables
_TOOL_REGISTRY: dict[str, callable] = {
    "search_web": search_web,
    "scrape_content": scrape_content,
    "summarize": summarize,
}


class MCPClient:
    """
    Lightweight MCP client that dispatches tool calls to registered Python functions.
    Mirrors the MCP tool call interface: call(tool_name, **kwargs) → dict.
    """

    def __init__(self):
        self._registry = _TOOL_REGISTRY
        logger.info(f"[MCPClient] Initialized with tools: {list(self._registry.keys())}")

    def list_tools(self) -> list[str]:
        """Return the names of all available tools."""
        return list(self._registry.keys())

    def call(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """
        Call a named MCP tool with keyword arguments.

        Args:
            tool_name: Name of the registered tool.
            **kwargs:  Arguments forwarded to the tool function.

        Returns:
            Tool output dict. Always includes an 'error' key (None on success).
        """
        if tool_name not in self._registry:
            logger.error(f"[MCPClient] Unknown tool: {tool_name}")
            return {"error": f"Tool '{tool_name}' not found. Available: {self.list_tools()}"}

        tool_fn = self._registry[tool_name]
        logger.info(f"[MCPClient] Calling '{tool_name}' with args: {list(kwargs.keys())}")

        try:
            result = tool_fn(**kwargs)
            logger.debug(f"[MCPClient] '{tool_name}' returned successfully.")
            return result
        except Exception as e:
            logger.exception(f"[MCPClient] Tool '{tool_name}' raised: {e}")
            return {"error": str(e)}

    def register_tool(self, name: str, fn: callable):
        """Dynamically register a new tool (used in Phase 3 for advanced tools)."""
        self._registry[name] = fn
        logger.info(f"[MCPClient] Registered new tool: {name}")
