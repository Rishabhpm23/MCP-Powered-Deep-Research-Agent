# 🔬 MCP-Powered Deep Research Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-FastMCP-FF6B35?style=for-the-badge&logo=anthropic&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C1C1C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)
![Tavily](https://img.shields.io/badge/Tavily-Search-00B4D8?style=for-the-badge)

**An AI-powered deep research agent that treats every research step as a modular MCP tool, orchestrated by a LangGraph agent for multi-hop reasoning.**

[Features](#features) • [Architecture](#architecture) • [Setup](#setup) • [Usage](#usage) • [Phases](#project-phases)

</div>

---

## ✨ Features

- **🔧 MCP-Native** — Every research step (search, scrape, summarize, cluster, report) is a discrete MCP tool with a strict JSON schema
- **🤖 LangGraph Orchestration** — Stateful agent with multi-hop reasoning, plan → execute → memory → finalize
- **🌐 Real Web Search** — Powered by Tavily API for up-to-date, ranked web results
- **📄 Smart Scraping** — BeautifulSoup-based content extractor that strips noise and returns clean text
- **🧠 LLM Summarization** — OpenAI GPT-4o produces focused, fact-dense summaries
- **📊 Structured Reports** — Outputs Markdown briefs, comparison tables, and insight reports
- **💾 Shared Memory** — Intermediate summaries are aggregated across hops
- **🖥️ React Dashboard** — Real-time tool-call trace viewer and report renderer *(Phase 4)*

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                 LangGraph Agent                      │
│                                                     │
│  ┌──────────┐    ┌─────────────┐    ┌────────────┐  │
│  │ Planner  │───▶│ Tool Caller │───▶│  Memory    │  │
│  └──────────┘    └──────┬──────┘    │ Aggregator │  │
│                         │           └─────┬──────┘  │
│                  JSON-RPC (MCP)           │         │
│                         │           ┌─────▼──────┐  │
│  ┌──────────────────────▼──────┐    │ Finalizer  │  │
│  │        MCP Server            │    └────────────┘  │
│  │                             │                   │
│  │  search_web    scrape_content│                   │
│  │  summarize     cluster_rank  │                   │
│  │  generate_report             │                   │
│  └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
Research Report (Markdown / JSON)
```

### MCP Tool Manifest

| Tool | Description | Key Params |
|---|---|---|
| `search_web` | Tavily-powered web search | `query`, `max_results`, `search_depth` |
| `scrape_content` | Clean text extraction from URL | `url`, `max_chars` |
| `summarize` | LLM-powered focused summarization | `text`, `focus`, `max_length` |
| `cluster_and_rank` | Semantic clustering + relevance ranking | `summaries`, `query`, `num_clusters` |
| `generate_report` | Structured Markdown/JSON report generation | `findings`, `query`, `format` |

---

## 📁 Project Structure

```
deep-research-agent/
├── mcp_server/
│   ├── server.py            # FastMCP server — registers all tools
│   ├── manifest.json        # Tool schemas and descriptions
│   └── tools/
│       ├── search_web.py    # Tavily search tool
│       ├── scrape_content.py# BeautifulSoup scraper
│       ├── summarize.py     # OpenAI summarizer
│       ├── cluster_and_rank.py  # (Phase 3)
│       └── generate_report.py   # (Phase 3)
├── agent/
│   ├── graph.py             # LangGraph state graph  (Phase 2)
│   ├── nodes.py             # Agent nodes            (Phase 2)
│   ├── memory.py            # Shared memory module   (Phase 2)
│   └── mcp_client.py        # MCP JSON-RPC client    (Phase 2)
├── frontend/                # React dashboard        (Phase 4)
├── tests/
│   └── test_tools.py        # Phase 1 CLI test runner
├── reports/                 # Generated research reports
├── .env.example
├── requirements.txt
├── docker-compose.yml       # (Phase 5)
└── README.md
```

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Tavily API key ([Free tier available](https://tavily.com))

### 1. Clone the Repository

```bash
git clone https://github.com/Rishabhpm23/deep-research-agent.git
cd deep-research-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
OPENAI_MODEL=gpt-4o
```

---

## 💻 Usage

### Phase 1 — Test Individual Tools

Run the full Phase 1 test suite:

```bash
python tests/test_tools.py
```

Test a specific tool:

```bash
# Search the web
python tests/test_tools.py --tool search_web --query "transformer architecture innovations 2024"

# Scrape a page
python tests/test_tools.py --tool scrape_content --url "https://arxiv.org/abs/2303.08774"

# Summarize (uses built-in sample text)
python tests/test_tools.py --tool summarize
```

### Start the MCP Server *(stdio mode)*

```bash
python -m mcp_server.server
```

---

## 🗺️ Project Phases

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Complete | MCP Server + Core Tools (search, scrape, summarize) |
| **Phase 2** | 🔲 Next | LangGraph Agent + MCP Integration + Shared Memory |
| **Phase 3** | 🔲 Planned | cluster_and_rank + generate_report tools |
| **Phase 4** | 🔲 Planned | React Frontend with real-time streaming |
| **Phase 5** | 🔲 Planned | Docker Compose + production polish |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| MCP Server | FastMCP (Python MCP SDK) |
| Agent Framework | LangGraph |
| LLM | OpenAI GPT-4o |
| Web Search | Tavily API |
| Scraping | httpx + BeautifulSoup4 |
| Clustering | scikit-learn |
| Frontend | React + TypeScript + Vite *(Phase 4)* |
| Containerization | Docker Compose *(Phase 5)* |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ by <a href="https://github.com/Rishabhpm23">Rishabhpm23</a>
</div>
