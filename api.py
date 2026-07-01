"""
api.py
──────
FastAPI backend that exposes the Deep Research Agent over HTTP.
Provides a streaming SSE endpoint for real-time progress updates
and a REST endpoint for completed report retrieval.

Run with: uvicorn api:app --reload --port 8080
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deep Research Agent API",
    description="MCP-Powered Deep Research Agent — REST & SSE API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ──────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    format: str = "markdown_brief"   # markdown_brief | comparison_table | insight_report
    max_hops: int = 5


class ResearchStatus(BaseModel):
    status: str
    hop_count: int
    search_result_count: int
    scraped_page_count: int
    summary_count: int
    tool_calls: int


# ── In-memory report store ─────────────────────────────────────────────────────
_report_store: dict[str, dict] = {}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── SSE Streaming Research Endpoint ──────────────────────────────────────────
@app.post("/research/stream")
async def stream_research(req: ResearchRequest):
    """
    Start a research session and stream real-time SSE events.
    Each event contains: type, data, timestamp.
    Event types: status, tool_call, step_complete, report, error
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    async def event_generator():
        try:
            # Import inside to avoid module-level LLM init errors
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from agent.memory import ResearchMemory
            from agent.graph import research_graph

            memory = ResearchMemory(query=req.query)
            initial_state = {
                "query": req.query,
                "plan": [],
                "current_step": 0,
                "messages": [],
                "memory": memory,
                "next_action": "plan",
                "error": "",
                "status": "Starting research...",
            }

            # Yield start event
            yield _sse_event("status", {
                "session_id": session_id,
                "message": "Research started",
                "query": req.query,
            })

            prev_hop = 0
            prev_tool_calls = 0

            # Stream graph events
            for event in research_graph.stream(initial_state, stream_mode="values"):
                mem = event.get("memory")
                status = event.get("status", "")
                current_step = event.get("current_step", 0)

                if status:
                    yield _sse_event("status", {
                        "message": status,
                        "step": current_step,
                        "plan_length": len(event.get("plan", [])),
                    })

                if mem:
                    # Emit new tool calls
                    if len(mem.tool_call_log) > prev_tool_calls:
                        for call in mem.tool_call_log[prev_tool_calls:]:
                            yield _sse_event("tool_call", {
                                "tool": call["tool"],
                                "hop": call["hop"],
                                "inputs": str(call["inputs"])[:200],
                                "output_preview": call["output_preview"][:300],
                            })
                        prev_tool_calls = len(mem.tool_call_log)

                    # Emit new hop
                    if mem.hop_count > prev_hop:
                        yield _sse_event("hop", {
                            "hop_number": mem.hop_count,
                            "summaries_so_far": len(mem.summaries),
                            "sources_found": len(mem.search_results),
                        })
                        prev_hop = mem.hop_count

                # Small async yield to not block
                await asyncio.sleep(0.01)

            # Final state
            final_mem = event.get("memory") if event else None
            report = final_mem.final_report if final_mem else ""

            # Store report
            _report_store[session_id] = {
                "query": req.query,
                "report": report,
                "stats": final_mem.to_dict() if final_mem else {},
                "tool_calls": final_mem.tool_call_log if final_mem else [],
                "created_at": datetime.now().isoformat(),
            }

            # Yield final report
            yield _sse_event("report", {
                "session_id": session_id,
                "report": report,
                "stats": final_mem.to_dict() if final_mem else {},
            })

            yield _sse_event("done", {"session_id": session_id, "message": "Research complete"})

        except Exception as e:
            logger.exception(f"[API] Research error: {e}")
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Report retrieval ───────────────────────────────────────────────────────────
@app.get("/report/{session_id}")
async def get_report(session_id: str):
    if session_id not in _report_store:
        raise HTTPException(status_code=404, detail="Report not found")
    return _report_store[session_id]


@app.get("/reports")
async def list_reports():
    return [
        {"session_id": sid, "query": data["query"], "created_at": data["created_at"]}
        for sid, data in _report_store.items()
    ]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _sse_event(event_type: str, data: dict) -> str:
    payload = json.dumps({"type": event_type, "data": data, "timestamp": datetime.now().isoformat()})
    return f"data: {payload}\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
