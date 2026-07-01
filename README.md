# 🔬 MCP-Powered Deep Research Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-FastMCP-FF6B35?style=for-the-badge&logo=anthropic&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-1C1C1C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)
![Tavily](https://img.shields.io/badge/Tavily-Search-00B4D8?style=for-the-badge)
![React](https://img.shields.io/badge/React-TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**An AI-powered deep research agent that treats every research step as a modular MCP tool, orchestrated by a stateful LangGraph agent for multi-hop reasoning.**

[Features](#-features) • [Architecture](#-architecture) • [MCP Tools](#-mcp-tool-manifest) • [Setup](#-setup) • [Usage](#-usage) • [Docker](#-docker) • [Phases](#-project-phases)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔧 **MCP-Native Tools** | Every research step is a discrete MCP tool with a strict JSON schema in `manifest.json` |
| 🤖 **LangGraph Orchestration** | Stateful 4-node agent: Planner → Tool Caller → Memory Aggregator → Finalizer |
| 🔁 **Multi-Hop Reasoning** | Agent loops over plan steps, accumulating context across up to 5 reasoning hops |
| 💾 **Shared Memory** | `ResearchMemory` object persists search results, scraped pages, and summaries across hops |
| 🌐 **Real Web Search** | Tavily API returns ranked, up-to-date results with relevance scores |
| 📄 **Smart Scraping** | BeautifulSoup prioritizes `<main>`, `<article>` content, strips ads/nav/boilerplate |
| 🧠 **Focused Summarization** | GPT-4o generates fact-dense summaries with configurable focus areas |
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
│  ├── summaries[]                                                  │
│  └── tool_call_log[]                                             │
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
│   ├── nodes.py                    # 4 agent nodes
│   ├── memory.py                   # ResearchMemory shared state
│   ├── mcp_client.py              # In-process MCP tool dispatcher
│   └── state.py                    # AgentState TypedDict
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

# With verbose tool trace
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

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **MCP Server** | FastMCP 2.x (Python MCP SDK) |
| **Agent Framework** | LangGraph 1.x (stateful StateGraph) |
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
| `error` | An error occurred |
| `done` | Session complete |

### `GET /report/{session_id}`
Retrieve a completed report by session ID.

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

⭐ Star this repo if it helped you learn about MCP agents!

</div>
