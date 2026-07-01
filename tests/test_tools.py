"""
test_tools.py
─────────────
Phase 1 CLI Test Runner — validates all three Phase 1 MCP tools:
  1. search_web    — Tavily API search
  2. scrape_content — URL scraping
  3. summarize      — OpenAI summarization

Usage:
    python tests/test_tools.py
    python tests/test_tools.py --tool search_web --query "AI agent frameworks 2024"
    python tests/test_tools.py --tool scrape_content --url "https://example.com"
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from mcp_server.tools.search_web import search_web
from mcp_server.tools.scrape_content import scrape_content
from mcp_server.tools.summarize import summarize

logging.basicConfig(level=logging.WARNING)  # Suppress info logs during tests
console = Console()


def _header(title: str):
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def test_search_web(query: str = "latest advances in large language models 2024"):
    _header("TEST: search_web")
    console.print(f"[dim]Query:[/dim] {query}\n")

    result = search_web(query=query, max_results=3, search_depth="basic")

    if result["error"]:
        console.print(f"[bold red]ERROR:[/bold red] {result['error']}")
        return False

    table = Table(title=f"Search Results for: '{query}'", show_lines=True)
    table.add_column("Score", style="green", width=7)
    table.add_column("Title", style="cyan", width=35)
    table.add_column("URL", style="blue", width=40)
    table.add_column("Snippet", width=50)

    for r in result["results"]:
        table.add_row(
            str(r["score"]),
            r["title"][:60],
            r["url"][:60],
            r["content"][:120] + "...",
        )

    console.print(table)
    console.print(f"\n[green]✓ Total results:[/green] {result['total_results']}\n")
    return True


def test_scrape_content(url: str = "https://en.wikipedia.org/wiki/Large_language_model"):
    _header("TEST: scrape_content")
    console.print(f"[dim]URL:[/dim] {url}\n")

    result = scrape_content(url=url, max_chars=2000)

    if result["error"]:
        console.print(f"[bold red]ERROR:[/bold red] {result['error']}")
        return False

    console.print(Panel(
        result["text"][:1500] + "\n[dim]...[/dim]",
        title=f"[cyan]{result['title']}[/cyan]",
        subtitle=f"[green]{result['char_count']} chars scraped[/green]",
        border_style="cyan",
    ))
    console.print(f"\n[green]✓ Scraped:[/green] {result['char_count']} chars from {url}\n")
    return True


def test_summarize():
    _header("TEST: summarize")
    sample_text = """
    Large language models (LLMs) are neural networks trained on massive text datasets
    using the transformer architecture. They learn statistical patterns in language to
    perform tasks like text generation, translation, summarization, and question answering.
    Models like GPT-4, Claude, and Gemini have demonstrated remarkable capabilities across
    diverse domains. The key innovation is the attention mechanism, which allows the model
    to weigh the importance of different words in context. Training these models requires
    enormous computational resources — often thousands of GPUs running for weeks. Despite
    their power, LLMs face challenges including hallucination, bias, and high inference costs.
    Recent research focuses on making them more efficient through techniques like quantization,
    distillation, and mixture-of-experts architectures.
    """
    console.print(f"[dim]Input text:[/dim] {len(sample_text)} chars\n")

    result = summarize(text=sample_text, focus="key challenges and recent solutions", max_length=80)

    if result["error"]:
        console.print(f"[bold red]ERROR:[/bold red] {result['error']}")
        return False

    console.print(Panel(
        result["summary"],
        title=f"[cyan]Summary[/cyan] [dim](focus: {result['focus']})[/dim]",
        subtitle=f"[green]{result['word_count']} words | model: {result['model']}[/green]",
        border_style="green",
    ))
    console.print(f"\n[green]✓ Summary generated:[/green] {result['word_count']} words\n")
    return True


def run_all_tests():
    """Run the full Phase 1 test suite."""
    console.print(Panel.fit(
        "[bold cyan]Deep Research Agent — Phase 1 Test Suite[/bold cyan]\n"
        "[dim]Testing: search_web | scrape_content | summarize[/dim]",
        border_style="cyan",
    ))

    results = {}
    results["search_web"] = test_search_web()
    results["scrape_content"] = test_scrape_content()
    results["summarize"] = test_summarize()

    # Summary
    _header("RESULTS")
    summary_table = Table(show_header=True)
    summary_table.add_column("Tool", style="cyan")
    summary_table.add_column("Status", style="bold")

    all_passed = True
    for tool, passed in results.items():
        status = "[green]✓ PASSED[/green]" if passed else "[red]✗ FAILED[/red]"
        summary_table.add_row(tool, status)
        if not passed:
            all_passed = False

    console.print(summary_table)
    if all_passed:
        console.print("\n[bold green]All Phase 1 tests passed! ✨[/bold green]\n")
    else:
        console.print("\n[bold red]Some tests failed. Check your .env API keys.[/bold red]\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test Phase 1 MCP tools individually or all at once.")
    parser.add_argument("--tool", choices=["search_web", "scrape_content", "summarize", "all"], default="all")
    parser.add_argument("--query", default="latest advances in large language models 2024")
    parser.add_argument("--url", default="https://en.wikipedia.org/wiki/Large_language_model")
    args = parser.parse_args()

    if args.tool == "all":
        run_all_tests()
    elif args.tool == "search_web":
        test_search_web(args.query)
    elif args.tool == "scrape_content":
        test_scrape_content(args.url)
    elif args.tool == "summarize":
        test_summarize()


if __name__ == "__main__":
    main()
