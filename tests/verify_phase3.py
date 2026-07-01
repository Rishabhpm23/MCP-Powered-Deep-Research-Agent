import sys; sys.path.insert(0, '.')
from mcp_server.tools.cluster_and_rank import cluster_and_rank
from mcp_server.tools.generate_report import generate_report
from agent.mcp_client import MCPClient

summaries = [
    'GPT-4 achieves state-of-the-art on reasoning tasks using transformer architecture.',
    'Claude excels at long-context with reduced hallucinations via Constitutional AI.',
    'Quantization allows large models to run efficiently on consumer hardware.',
    'RAG systems reduce hallucinations by grounding LLM outputs in retrieved documents.',
]

result = cluster_and_rank(summaries=summaries, query='LLM improvements 2024', num_clusters=2)
nc = result['num_clusters']
err = result['error']
print(f"cluster_and_rank: {nc} clusters, error={err}")
for c in result['clusters']:
    cid = c['cluster_id']
    rel = c['relevance_score']
    theme = c['theme']
    print(f"  Cluster {cid}: relevance={rel}, theme={theme}")

mcp = MCPClient()
tools = mcp.list_tools()
print(f"MCPClient tools: {tools}")
print("Phase 3 imports OK")
