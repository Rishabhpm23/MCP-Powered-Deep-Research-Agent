"""
run_agent.py
────────────
CLI entry point for Zetabot — MCP-Powered Deep Research Agent.
Accepts a research query and runs it through the full LangGraph pipeline.

Usage:
    python run_agent.py --query "What are the latest breakthroughs in AI reasoning?"
    python run_agent.py --query "..." --format markdown_brief --output ./reports/
    python run_agent.py --query "..." --verbose
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint

# Set up project root on path
sys.path.insert(0, str(Path(__file__).parent))

from agent.memory import ResearchMemory
from agent.graph import research_graph

console = Console()


def save_report(report: str, query: str, output_dir: str) -> str:
    """Save the research report to a Markdown file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in query[:40])
    filename = f"{timestamp}_{safe_query}.md"
    filepath = Path(output_dir) / filename
    filepath.write_text(report, encoding="utf-8")
    return str(filepath)


def run_research(query: str, verbose: bool = False, output_dir: str = "./reports") -> str:
    """
    Run the full research agent for a given query.
    Returns the final Markdown report as a string.
    """
    console.print(Panel.fit(
        f"[bold cyan]⚡ Zetabot[/bold cyan]\n"
        f"[dim]Query:[/dim] {query}",
        border_style="cyan",
    ))

    # Initialize memory
    memory = ResearchMemory(query=query)

    initial_state = {
        "query": query,
        "plan": [],
        "current_step": 0,
        "messages": [],
        "memory": memory,
        "next_action": "plan",
        "errors": [],          # Improvement #3 — accumulated list, not single string
        "status": "Starting...",
    }

    final_state = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Researching...", total=None)

        # Stream through graph events
        for event in research_graph.stream(initial_state, stream_mode="values"):
            status = event.get("status", "")
            if status:
                progress.update(task, description=f"[cyan]{status}")
                if verbose:
                    console.print(f"  [dim]▶ {status}[/dim]")
            final_state = event

    if not final_state:
        console.print("[red]Error: No output from agent.[/red]")
        return ""

    mem: ResearchMemory = final_state.get("memory")
    report = mem.final_report if mem else ""

    # ── Print summary table ────────────────────────────────────────────────────
    if mem:
        table = Table(title="Research Summary", show_header=True, border_style="cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in mem.to_dict().items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)

    # ── Print tool call trace if verbose ──────────────────────────────────────
    if verbose and mem and mem.tool_call_log:
        console.rule("[bold yellow]Tool Call Trace[/bold yellow]")
        for i, call in enumerate(mem.tool_call_log, 1):
            console.print(
                f"  [yellow]{i}.[/yellow] [cyan]{call['tool']}[/cyan] "
                f"[dim](hop {call['hop']})[/dim] → {call['output_preview'][:100]}..."
            )

    # ── Print accumulated error log if verbose and any errors occurred ─────────
    if verbose and mem and mem.error_log:
        console.rule("[bold red]Error Log (Accumulated)[/bold red]")
        for i, err in enumerate(mem.error_log, 1):
            console.print(f"  [red]{i}.[/red] {err}")

    # ── Render the report ──────────────────────────────────────────────────────
    if report:
        console.rule("[bold green]Research Report[/bold green]")
        console.print(Markdown(report))

        # Save to file
        filepath = save_report(report, query, output_dir)
        console.print(f"\n[green]✓ Report saved to:[/green] {filepath}\n")
    else:
        console.print("[red]No report generated.[/red]")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Zetabot — MCP-Powered Deep Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py --query "Latest breakthroughs in quantum computing 2024"
  python run_agent.py --query "Compare GPT-4 vs Claude 3" --verbose
  python run_agent.py --query "AI safety research" --output ./reports/
        """,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="The research query to investigate.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.getenv("REPORTS_OUTPUT_DIR", "./reports"),
        help="Directory to save the generated report (default: ./reports).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed tool call trace and step-by-step progress.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Python logging level (default: WARNING).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    run_research(query=args.query, verbose=args.verbose, output_dir=args.output)


if __name__ == "__main__":
    main()
