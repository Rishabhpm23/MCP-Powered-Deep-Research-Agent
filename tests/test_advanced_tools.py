"""
test_advanced_tools.py
──────────────────────
Phase 3 test runner — validates cluster_and_rank and generate_report tools.
Run with: python tests/test_advanced_tools.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

logging.basicConfig(level=logging.WARNING)
console = Console()

from mcp_server.tools.cluster_and_rank import cluster_and_rank
from mcp_server.tools.generate_report import generate_report


SAMPLE_SUMMARIES = [
    "GPT-4o achieves state-of-the-art performance on reasoning benchmarks including MMLU and HumanEval, with multimodal capabilities for text and images.",
    "Claude 3.5 Sonnet excels at long-context understanding up to 200K tokens and demonstrates strong coding ability with reduced hallucination rates.",
    "LLM hallucination remains a critical challenge; techniques like RAG, chain-of-thought, and fine-tuning reduce but don't eliminate fabrication.",
    "Mixture-of-Experts (MoE) architectures like Mixtral activate only a subset of parameters per token, enabling efficient scaling of model capacity.",
    "Quantization and distillation allow large models to run on consumer hardware, with 4-bit quantized models achieving near full-precision performance.",
    "Reinforcement Learning from Human Feedback (RLHF) and Constitutional AI are the main alignment techniques used to make LLMs safe and helpful.",
    "AI agents built on LLMs can plan, use tools, and complete multi-step tasks, though reliability decreases with task complexity.",
]

SAMPLE_QUERY = "What are the latest advances and challenges in large language models?"


def _header(title: str):
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def test_cluster_and_rank():
    _header("TEST: cluster_and_rank")
    console.print(f"[dim]Summaries:[/dim] {len(SAMPLE_SUMMARIES)} | [dim]Query:[/dim] {SAMPLE_QUERY[:60]}...\n")

    result = cluster_and_rank(summaries=SAMPLE_SUMMARIES, query=SAMPLE_QUERY, num_clusters=3)

    if result["error"]:
        console.print(f"[bold red]ERROR:[/bold red] {result['error']}")
        return False

    for cluster in result["clusters"]:
        console.print(Panel(
            "\n".join(f"  • {s[:100]}..." for s in cluster["summaries"]),
            title=f"[cyan]Cluster {cluster['cluster_id']}[/cyan] — Relevance: [green]{cluster['relevance_score']:.4f}[/green]",
            subtitle=f"[dim]Theme: {cluster['theme']}[/dim]",
            border_style="cyan",
        ))

    console.print(f"\n[green]✓ Clustered {result['total_summaries']} summaries into {result['num_clusters']} groups.[/green]\n")
    return True


def test_generate_report_markdown():
    _header("TEST: generate_report (markdown_brief)")
    result = generate_report(
        findings=SAMPLE_SUMMARIES[:4],
        query=SAMPLE_QUERY,
        format="markdown_brief",
    )
    if result["error"]:
        console.print(f"[bold red]ERROR:[/bold red] {result['error']}")
        return False
    console.print(Markdown(result["report"]))
    console.print(f"\n[green]✓ markdown_brief generated: {result['word_count']} words.[/green]\n")
    return True


def test_generate_report_insight():
    _header("TEST: generate_report (insight_report)")
    result = generate_report(
        findings=SAMPLE_SUMMARIES,
        query=SAMPLE_QUERY,
        format="insight_report",
    )
    if result["error"]:
        console.print(f"[bold red]ERROR:[/bold red] {result['error']}")
        return False
    console.print(Markdown(result["report"][:2000] + "\n\n[dim]...(truncated)[/dim]"))
    console.print(f"\n[green]✓ insight_report generated: {result['word_count']} words.[/green]\n")
    return True


def run_all():
    console.print(Panel.fit(
        "[bold cyan]Deep Research Agent — Phase 3 Test Suite[/bold cyan]\n"
        "[dim]Testing: cluster_and_rank | generate_report[/dim]",
        border_style="cyan",
    ))

    results = {
        "cluster_and_rank": test_cluster_and_rank(),
        "generate_report (markdown)": test_generate_report_markdown(),
        "generate_report (insight)": test_generate_report_insight(),
    }

    _header("RESULTS")
    table = Table()
    table.add_column("Tool", style="cyan")
    table.add_column("Status", style="bold")
    all_passed = True
    for tool, passed in results.items():
        table.add_row(tool, "[green]✓ PASSED[/green]" if passed else "[red]✗ FAILED[/red]")
        if not passed:
            all_passed = False
    console.print(table)
    if all_passed:
        console.print("\n[bold green]All Phase 3 tests passed! ✨[/bold green]\n")
    else:
        console.print("\n[bold red]Some tests failed. Check your .env API keys.[/bold red]\n")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
