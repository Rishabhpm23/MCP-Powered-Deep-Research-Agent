# 🔬 MCP-Powered Deep Research Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-FastMCP-FF6B35?style=for-the-badge&logo=anthropic&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-1C1C1C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)
![Tavily](https://img.shields.io/badge/Tavily-Search-00B4D8?style=for-the-badge)
![React](https://img.shields.io/badge/React-TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**An AI-powered deep research agent that treats every research step as a modular MCP tool, orchestrated by a stateful LangGraph agent with production-grade context engineering for multi-hop reasoning.**

[Features](#-features) • [Architecture](#-architecture) • [Context Engineering](#-context-engineering) • [MCP Tools](#-mcp-tool-manifest) • [Setup](#-setup) • [Usage](#-usage) • [Docker](#-docker) • [Phases](#-project-phases)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔧 **MCP-Native Tools** | Every research step is a discrete MCP tool with a strict JSON schema in `manifest.json` |
| 🤖 **LangGraph Orchestration** | Stateful 4-node agent: Planner → Tool Caller → Memory Aggregator → Finalizer |
| 🔁 **Multi-Hop Reasoning** | Agent loops over plan steps, accumulating context across up to 5 reasoning hops |
| 🧠 **Context Engineering** | `ContextBuilder` provides token-budgeted, relevance-ranked, dynamic context per LLM call |
| 💾 **Shared Memory** | `ResearchMemory` persists search results, scraped pages, summaries, and error logs across hops |
| 🌐 **Real Web Search** | Tavily API returns ranked, up-to-date results with relevance scores |
| 📄 **Smart Scraping** | BeautifulSoup prioritizes `<main>`, `<article>` content, strips ads/nav/boilerplate |
| ✍️ **Focused Summarization** | GPT-4o generates fact-dense summaries with configurable focus areas and confidence scores |
| 🗂️ **Semantic Clustering** | TF-IDF + K-Means groups findings; cosine similarity ranks by query relevance |
| 📊 **Structured Reports** | 3 output formats: Markdown Brief, Comparison Table, Insight Report |
| ⚡ **Real-Time Streaming** | FastAPI SSE streams tool calls, hops, and status live to the React dashboard |
| 🖥️ **React Dashboard** | Dark glassmorphism UI with trace viewer, stats, and Markdown report renderer |
| 🐳 **Docker Ready** | Full Docker Compose stack for one-command deployment |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        React Frontend (Vite)                      │
│  Query Input → SSE Stream → Tool Trace → Stats → Report Viewer   │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP POST + SSE (port 8080)
┌───────────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend (api.py)                        │
│  /research/stream  /report/{id}  /reports  /health               │
└───────────────────────────┬──────────────────────────────────────┘
                            │ Python calls
┌───────────────────────────▼──────────────────────────────────────┐
│                   LangGraph Agent (agent/)                         │
│                                                                  │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────────────┐   │
│  │ Planner  │───▶│ Tool Caller │───▶│  Memory Aggregator    │   │
│  │  (LLM)   │    │  (MCP loop) │    │  (auto-summarize)     │   │
│  └──────────┘    └──────┬──────┘    └──────────┬────────────┘   │
│       ↑              loop│                      │                 │
│       └──────────────────┘                      ▼                 │
│                                         ┌─────────────┐          │
│  ResearchMemory (shared across hops)    │  Finalizer  │          │
│  ├── search_results[]                   │  (LLM report)│         │
│  ├── scraped_pages[]                    └─────────────┘          │
│  ├── summaries[] + confidence_scores[]     ↑                     │
│  ├── error_log[]                           │                     │
│  └── tool_call_log[]           ContextBuilder (agent/context.py) │
└──────────────────────────────────────────────────────────────────┘
                            │ MCPClient dispatch
┌───────────────────────────▼──────────────────────────────────────┐
│                   MCP Server (mcp_server/)                         │
│                                                                  │
│  🔍 search_web        🕷️ scrape_content    ✍️ summarize          │
│  🗂️ cluster_and_rank  📋 generate_report                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Context Engineering

A core principle of this agent is **context engineering** — ensuring every LLM call receives precisely the right information: no more, no less. This is handled by `agent/context.py` (`ContextBuilder`), which replaces naive, ad-hoc prompt assembly.

### Problems Solved

| # | Problem (Before) | Solution (After) |
|---|---|---|
| 1 | Tool caller only saw the last 500 chars of the most recent scraped page | `build_tool_caller_context()` injects full tool history, all found URLs, latest summary preview, and accumulated errors |
| 2 | Finalizer received an unbounded dump of all summaries (no token cap) | `build_finalizer_context()` keyword-scores summaries by query overlap and selects top-N within a 5 000-token budget |
| 3 | All system prompts were static strings with no session awareness | All nodes use dynamic prompts that inject hop budget, tool list, progress counts, and strategic nudges |
| 4 | `messages` list was write-only; no conversational continuity | Tool caller injects `state["messages"][-4:]` into each LLM call |
| 5 | A single `error: str` field was silently overwritten each hop | `errors: list[str]` in `AgentState` uses a LangGraph reducer that appends — full error history preserved |
| 6 | No per-hop quality signal for summaries | `confidence_scores: list[float]` recorded alongside each summary; finalizer can weight by confidence |

### How It Works

```
ContextBuilder(query, max_hops, max_tokens=5000)
│
├── build_planner_system_prompt(available_tools)
│     → Injects MAX_HOPS + live tool list into planner's system message
│
├── build_tool_caller_system_prompt(memory)
│     → Injects hops_remaining, progress counts, strategic nudge
│       ("PRIORITIZE summarization" when hops ≤ 2)
│
├── build_tool_caller_context(memory)
│     → Rich context block: tool history (last 5 hops, 150-char previews),
│       all found URLs (dedup avoidance), latest summary, error log
│
├── build_finalizer_system_prompt(memory)
│     → Adapts depth/persona to data richness
│       (comprehensive if ≥8 data points; honest about gaps if <3)
│
└── build_finalizer_context(memory)
      → Scores summaries by keyword overlap with query
      → Selects top summaries within 5 000-token char budget
      → Returns curated sources block (top-5 by search rank)
```

### Key Files Changed

| File | Change |
|---|---|
| `agent/context.py` *(new)* | `ContextBuilder` — all context assembly logic |
| `agent/memory.py` | Added `error_log`, `confidence_scores`, `get_relevant_summaries()`, `get_top_sources()` |
| `agent/state.py` | `error: str` → `errors: Annotated[list[str], _accumulate_errors]` |
| `agent/nodes.py` | All 4 nodes use `ContextBuilder`; conversational memory; confidence scores |
| `api.py` | `initial_state` uses `errors: []`; report store exposes `error_log` |
| `run_agent.py` | Verbose mode prints accumulated error log |

---

## 🔧 MCP Tool Manifest

All tools are defined in `mcp_server/manifest.json` with full JSON schemas.

| Tool | Description | Key Parameters |
|---|---|---|
| `search_web` | Tavily API web search | `query`, `max_results`, `search_depth` |
| `scrape_content` | BeautifulSoup content extraction | `url`, `max_chars` |
| `summarize` | GPT-4o focused summarization | `text`, `focus`, `max_length` |
| `cluster_and_rank` | TF-IDF + K-Means + cosine ranking | `summaries`, `query`, `num_clusters` |
| `generate_report` | Structured report in 3 formats | `findings`, `query`, `format` |

---

## 📁 Project Structure

```
MCP-Powered-Deep-Research-Agent/
│
├── mcp_server/                     # FastMCP server
│   ├── server.py                   # Registers all tools via @mcp.tool()
│   ├── manifest.json               # Tool schemas (JSON Schema)
│   └── tools/
│       ├── search_web.py           # Tavily search
│       ├── scrape_content.py       # BeautifulSoup scraper
│       ├── summarize.py            # OpenAI summarizer
│       ├── cluster_and_rank.py     # TF-IDF + K-Means clustering
│       └── generate_report.py      # 3-format report generator
│
├── agent/                          # LangGraph orchestration
│   ├── graph.py                    # StateGraph with conditional routing
│   ├── nodes.py                    # 4 agent nodes (context-engineering-aware)
│   ├── memory.py                   # ResearchMemory: results, summaries, error_log, confidence_scores
│   ├── context.py                  # ContextBuilder: token-budgeted, relevance-ranked prompts
│   ├── mcp_client.py               # In-process MCP tool dispatcher
│   └── state.py                    # AgentState TypedDict (accumulated errors list)
│
├── frontend/                       # React + TypeScript + Vite
│   ├── src/
│   │   ├── App.tsx                 # Main component (SSE + UI)
│   │   └── index.css               # Design system (glassmorphism)
│   ├── Dockerfile.frontend
│   └── nginx.conf                  # SPA routing + API proxy
│
├── tests/
│   ├── test_tools.py               # Phase 1 CLI test runner
│   └── test_advanced_tools.py      # Phase 3 test runner
│
├── examples/
│   └── sample_report_llm_reasoning.md   # Example agent output
│
├── api.py                          # FastAPI backend (SSE streaming)
├── run_agent.py                    # CLI entry point
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
├── .env.example
└── README.md
```

---

## 🚀 Setup

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- OpenAI API key → [platform.openai.com](https://platform.openai.com)
- Tavily API key → [tavily.com](https://tavily.com) *(free tier available)*

### 1. Clone & Configure

```bash
git clone https://github.com/Rishabhpm23/MCP-Powered-Deep-Research-Agent.git
cd MCP-Powered-Deep-Research-Agent

cp .env.example .env
# Edit .env and add your API keys
```

### 2. Python Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Frontend

```bash
cd frontend
npm install
cd ..
```

---

## 💻 Usage

### Option A — CLI Research Agent

Run a full research query end-to-end from the terminal:

```bash
# Basic query
python run_agent.py --query "What are the latest advances in AI reasoning?"

# With verbose tool trace + accumulated error log
python run_agent.py --query "Compare GPT-4o vs Claude 3.5 Sonnet" --verbose

# Save report to custom directory
python run_agent.py --query "AI safety research 2024" --output ./my-reports/
```

### Option B — Full Stack (API + Frontend)

**Terminal 1 — Start the FastAPI backend:**
```bash
uvicorn api:app --reload --port 8080
```

**Terminal 2 — Start the React frontend:**
```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Option C — Test Individual Tools

```bash
# Test all Phase 1 tools
python tests/test_tools.py

# Test a specific tool
python tests/test_tools.py --tool search_web --query "transformer architecture"
python tests/test_tools.py --tool scrape_content --url "https://en.wikipedia.org/wiki/GPT-4"

# Test Phase 3 advanced tools
python tests/test_advanced_tools.py
```

---

## 🐳 Docker

Run the complete stack with a single command:

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build
```

| Service | URL |
|---|---|
| React Frontend | [http://localhost:3000](http://localhost:3000) |
| FastAPI Backend | [http://localhost:8080](http://localhost:8080) |
| API Docs | [http://localhost:8080/docs](http://localhost:8080/docs) |

**Stop:**
```bash
docker-compose down
```

---

## ⚙️ Configuration

All settings are controlled via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `TAVILY_API_KEY` | — | **Required.** Tavily search API key |
| `OPENAI_MODEL` | `gpt-4o` | LLM model to use |
| `MAX_HOPS` | `5` | Max reasoning hops per query |
| `MAX_SEARCH_RESULTS` | `5` | Results per `search_web` call |
| `MAX_SCRAPE_CHARS` | `8000` | Chars returned per `scrape_content` |
| `NUM_CLUSTERS` | `3` | Clusters for `cluster_and_rank` |
| `REPORT_FORMAT` | `markdown_brief` | Default output format |
| `REPORTS_OUTPUT_DIR` | `./reports` | Where CLI saves reports |

---

## 📊 Report Formats

| Format | Description | Best For |
|---|---|---|
| `markdown_brief` | Executive summary + key findings + analysis + sources | General research |
| `insight_report` | Detailed narrative with trends, challenges, recommendations | Deep dives |
| `comparison_table` | Structured table comparing sources/approaches | Competitive analysis |

See [`examples/sample_report_llm_reasoning.md`](examples/sample_report_llm_reasoning.md) for a full example output.

---

## 🗺️ Project Phases

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Complete | MCP Server + `search_web`, `scrape_content`, `summarize` tools |
| **Phase 2** | ✅ Complete | LangGraph Agent + MCP client + shared memory + CLI |
| **Phase 3** | ✅ Complete | `cluster_and_rank` + `generate_report` advanced tools |
| **Phase 4** | ✅ Complete | React dashboard with real-time SSE streaming |
| **Phase 5** | ✅ Complete | Docker Compose + final README + example outputs |
| **Phase 6** | ✅ Complete | Context Engineering — `ContextBuilder`, dynamic prompts, token-budgeted context, accumulated error log, confidence scoring |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **MCP Server** | FastMCP 2.x (Python MCP SDK) |
| **Agent Framework** | LangGraph 1.x (stateful StateGraph) |
| **Context Engineering** | Custom `ContextBuilder` (token budgeting, relevance ranking, dynamic prompts) |
| **LLM** | OpenAI GPT-4o (configurable) |
| **Web Search** | Tavily Python SDK |
| **Web Scraping** | httpx + BeautifulSoup4 + lxml |
| **Clustering** | scikit-learn (TF-IDF, K-Means, cosine similarity) |
| **API Backend** | FastAPI + uvicorn + SSE |
| **Frontend** | React 19 + TypeScript + Vite 6 |
| **UI** | Vanilla CSS (glassmorphism, dark mode) |
| **Containerization** | Docker Compose |

---

## 🔌 API Reference

### `POST /research/stream`
Start a research session. Returns a Server-Sent Events stream.

**Request body:**
```json
{
  "query": "Your research question",
  "format": "markdown_brief",
  "max_hops": 5
}
```

**SSE Event types:**

| Type | Description |
|---|---|
| `status` | Agent status update (planning, searching, etc.) |
| `tool_call` | A specific MCP tool was invoked |
| `hop` | A reasoning hop completed |
| `report` | Final report ready (includes full Markdown) |
| `error` | A streaming-level error occurred |
| `done` | Session complete |

### `GET /report/{session_id}`
Retrieve a completed report by session ID. Response includes `report`, `stats`, `tool_calls`, and `errors` (accumulated error log).

### `GET /reports`
List all reports from the current session.

### `GET /health`
Health check endpoint.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ by [Rishabhpm23](https://github.com/Rishabhpm23)

⭐ Star this repo if it helped you learn about MCP agents and context engineering!

</div>
