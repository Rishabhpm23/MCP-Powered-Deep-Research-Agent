import { useState, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { Zap, Copy, Check, FileText } from 'lucide-react';
import './index.css';

// ── Types ─────────────────────────────────────────────────────────────────────
interface TraceItem {
  type: 'status' | 'tool_call' | 'hop' | 'error';
  tool?: string;
  message?: string;
  hop?: number;
  inputs?: string;
  output_preview?: string;
  timestamp: string;
}

interface Stats {
  query: string;
  hop_count: number;
  search_result_count: number;
  scraped_page_count: number;
  summary_count: number;
  tool_calls: number;
}

interface HistoryEntry {
  query: string;
  time: string;
  report: string;
}

// ── Constants ─────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8080';

const TOOLS = [
  { name: 'search_web',       icon: '🔍', desc: 'Tavily web search' },
  { name: 'scrape_content',   icon: '🕷️', desc: 'BeautifulSoup scraper' },
  { name: 'summarize',        icon: '✍️', desc: 'GPT-4o summarizer' },
  { name: 'cluster_and_rank', icon: '🗂️', desc: 'TF-IDF clustering' },
  { name: 'generate_report',  icon: '📋', desc: 'Structured reports' },
];

const EXAMPLE_QUERIES = [
  'What are the latest advances in LLM reasoning?',
  'Compare GPT-4o vs Claude 3.5 Sonnet capabilities',
  'State of AI agents and multi-hop reasoning in 2024',
  'What are the main challenges in AI safety research?',
];

function getToolIcon(tool: string) {
  const icons: Record<string, string> = {
    search_web: '🔍', scrape_content: '🕷️', summarize: '✍️',
    cluster_and_rank: '🗂️', generate_report: '📋',
  };
  return icons[tool] || '⚡';
}

function getTraceClass(tool?: string) {
  const classes: Record<string, string> = {
    search_web: 'search', scrape_content: 'scrape',
    summarize: 'summarize', cluster_and_rank: 'cluster',
    generate_report: 'report',
  };
  return classes[tool || ''] || 'status';
}

// ── App Component ──────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery] = useState('');
  const [format, setFormat] = useState('markdown_brief');
  const [isResearching, setIsResearching] = useState(false);
  const [trace, setTrace] = useState<TraceItem[]>([]);
  const [report, setReport] = useState('');
  const [stats, setStats] = useState<Stats | null>(null);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [error, setError] = useState('');
  const traceRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const scrollTrace = () => {
    setTimeout(() => {
      if (traceRef.current) {
        traceRef.current.scrollTop = traceRef.current.scrollHeight;
      }
    }, 50);
  };

  const startResearch = useCallback(async (q?: string) => {
    const activeQuery = q || query;
    if (!activeQuery.trim() || isResearching) return;

    setIsResearching(true);
    setTrace([]);
    setReport('');
    setStats(null);
    setError('');

    abortRef.current = new AbortController();

    try {
      const response = await fetch(`${API_BASE}/research/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: activeQuery, format }),
        signal: abortRef.current.signal,
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));
            const { type, data, timestamp } = event;

            if (type === 'status') {
              setTrace(prev => [...prev, { type: 'status', message: data.message, timestamp }]);
              scrollTrace();
            } else if (type === 'tool_call') {
              setTrace(prev => [...prev, { type: 'tool_call', tool: data.tool, inputs: data.inputs, output_preview: data.output_preview, hop: data.hop, timestamp }]);
              scrollTrace();
            } else if (type === 'hop') {
              setTrace(prev => [...prev, { type: 'hop', message: `Hop ${data.hop_number} complete — ${data.summaries_so_far} summaries, ${data.sources_found} sources`, timestamp }]);
              scrollTrace();
            } else if (type === 'report') {
              setReport(data.report);
              setStats(data.stats);
              if (data.report) {
                setHistory(prev => [{ query: activeQuery, time: new Date().toLocaleTimeString(), report: data.report }, ...prev.slice(0, 9)]);
              }
            } else if (type === 'error') {
              setError(data.message);
            }
          } catch { /* skip malformed */ }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Connection failed. Make sure the API server is running on port 8080.');
      }
    } finally {
      setIsResearching(false);
    }
  }, [query, format, isResearching]);

  const stopResearch = () => {
    abortRef.current?.abort();
    setIsResearching(false);
  };

  const copyReport = () => {
    navigator.clipboard.writeText(report);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-icon">🔬</div>
          <div>
            <div className="header-title">Deep Research Agent</div>
            <div className="header-subtitle">MCP-Powered Multi-Hop Research</div>
          </div>
        </div>
        <div className="header-badge">
          <div className="badge-dot" />
          <span>5 MCP Tools Active</span>
        </div>
      </header>

      {/* Main grid */}
      <div className="main">
        {/* Left column */}
        <div>
          {/* Query Panel */}
          <div className="query-panel">
            <div className="panel-label">Research Query</div>
            <textarea
              id="research-query"
              className="query-textarea"
              placeholder="What would you like to research? E.g. 'What are the latest advances in AI reasoning and what are the main remaining challenges?'"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) startResearch(); }}
              rows={4}
              disabled={isResearching}
            />

            {/* Example queries */}
            <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {EXAMPLE_QUERIES.map(q => (
                <button
                  key={q}
                  onClick={() => { setQuery(q); }}
                  style={{
                    padding: '4px 10px', fontSize: 11, background: 'var(--bg-glass)',
                    border: '1px solid var(--border-subtle)', borderRadius: 20,
                    color: 'var(--text-muted)', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                    transition: 'all 150ms ease',
                  }}
                  onMouseEnter={e => { (e.target as HTMLElement).style.color = 'var(--text-primary)'; (e.target as HTMLElement).style.borderColor = 'var(--border-normal)'; }}
                  onMouseLeave={e => { (e.target as HTMLElement).style.color = 'var(--text-muted)'; (e.target as HTMLElement).style.borderColor = 'var(--border-subtle)'; }}
                >
                  {q.slice(0, 40)}...
                </button>
              ))}
            </div>

            <div className="query-options">
              <select
                id="report-format"
                className="format-select"
                value={format}
                onChange={e => setFormat(e.target.value)}
                disabled={isResearching}
              >
                <option value="markdown_brief">📄 Markdown Brief</option>
                <option value="insight_report">💡 Insight Report</option>
                <option value="comparison_table">📊 Comparison Table</option>
              </select>

              {isResearching ? (
                <button className="submit-btn" onClick={stopResearch} style={{ background: 'linear-gradient(135deg, #ef4444, #dc2626)' }}>
                  <span className="spinner" /> Stop
                </button>
              ) : (
                <button
                  id="start-research-btn"
                  className="submit-btn"
                  onClick={() => startResearch()}
                  disabled={!query.trim()}
                >
                  <Zap size={14} />
                  Research  <span style={{ fontSize: 10, opacity: 0.7, marginLeft: 4 }}>⌘↵</span>
                </button>
              )}
            </div>
          </div>

          {/* Stats Bar */}
          {stats && (
            <div className="stats-bar">
              {[
                { label: 'Hops', value: stats.hop_count },
                { label: 'Sources', value: stats.search_result_count },
                { label: 'Pages Scraped', value: stats.scraped_page_count },
                { label: 'Tool Calls', value: stats.tool_calls },
              ].map(s => (
                <div key={s.label} className="stat-card">
                  <div className="stat-value">{s.value}</div>
                  <div className="stat-label">{s.label}</div>
                </div>
              ))}
            </div>
          )}

          {/* Trace Panel */}
          {(trace.length > 0 || isResearching) && (
            <div className="trace-panel">
              <div className="trace-header">
                <div className="trace-title">
                  {isResearching && <span className="spinner" style={{ marginRight: 8 }} />}
                  Tool Call Trace
                </div>
                <div className="trace-count">{trace.length} events</div>
              </div>
              <div className="trace-list" ref={traceRef}>
                {trace.map((item, i) => (
                  <div key={i} className="trace-item">
                    <div className={`trace-item-icon ${getTraceClass(item.tool)}`}>
                      {item.tool ? getToolIcon(item.tool) : '●'}
                    </div>
                    <div className="trace-item-content">
                      <div className="trace-item-tool">
                        {item.tool || (item.type === 'hop' ? '↩ Hop Complete' : 'Status')}
                      </div>
                      <div className="trace-item-desc">
                        {item.output_preview || item.message || ''}
                      </div>
                    </div>
                  </div>
                ))}
                {isResearching && (
                  <div className="trace-item">
                    <div className="trace-item-icon status">⋯</div>
                    <div className="trace-item-content">
                      <div className="trace-item-tool">Processing...</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{ marginTop: 16, padding: '14px 18px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 12, fontSize: 13, color: '#fca5a5' }}>
              ⚠️ {error}
            </div>
          )}

          {/* Report Viewer */}
          {report ? (
            <div className="report-panel">
              <div className="report-header">
                <div className="report-title">
                  <FileText size={14} /> Research Report
                </div>
                <button className="copy-btn" onClick={copyReport}>
                  {copied ? <><Check size={12} /> Copied!</> : <><Copy size={12} /> Copy</>}
                </button>
              </div>
              <div className="report-content">
                <ReactMarkdown>{report}</ReactMarkdown>
              </div>
            </div>
          ) : !isResearching && trace.length === 0 && (
            <div className="empty-state">
              <div className="empty-state-icon">🔬</div>
              <div className="empty-state-title">Ready to Research</div>
              <div className="empty-state-text">
                Enter a research query above and click Research. The agent will search the web,
                scrape sources, and generate a structured report.
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="sidebar">
          {/* MCP Tools */}
          <div className="sidebar-card">
            <div className="sidebar-card-title">MCP Tools</div>
            <div className="tools-grid">
              {TOOLS.map(tool => (
                <div key={tool.name} className="tool-chip">
                  <div className="tool-chip-icon">{tool.icon}</div>
                  <div>
                    <div className="tool-chip-name">{tool.name}</div>
                    <div className="tool-chip-desc">{tool.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Research History */}
          {history.length > 0 && (
            <div className="sidebar-card">
              <div className="sidebar-card-title">Recent Queries</div>
              <div className="history-list">
                {history.map((h, i) => (
                  <div key={i} className="history-item" onClick={() => { setQuery(h.query); setReport(h.report); }}>
                    <div className="history-query">{h.query}</div>
                    <div className="history-time">{h.time}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* How it works */}
          <div className="sidebar-card">
            <div className="sidebar-card-title">How It Works</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                { icon: '🧠', label: 'Plan', desc: 'LLM breaks query into steps' },
                { icon: '🔍', label: 'Search', desc: 'Tavily retrieves top sources' },
                { icon: '🕷️', label: 'Scrape', desc: 'BeautifulSoup extracts text' },
                { icon: '✍️', label: 'Summarize', desc: 'GPT-4o generates summaries' },
                { icon: '🗂️', label: 'Cluster', desc: 'TF-IDF groups findings' },
                { icon: '📋', label: 'Report', desc: 'Structured Markdown output' },
              ].map((step) => (
                <div key={step.label} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 28, height: 28, background: 'var(--bg-glass)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, flexShrink: 0, border: '1px solid var(--border-subtle)' }}>
                    {step.icon}
                  </div>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{step.label}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{step.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        MCP-Powered Deep Research Agent · Built with LangGraph + FastMCP ·{' '}
        <a href="https://github.com/Rishabhpm23/MCP-Powered-Deep-Research-Agent" target="_blank" rel="noreferrer">GitHub</a>
      </footer>
    </div>
  );
}
