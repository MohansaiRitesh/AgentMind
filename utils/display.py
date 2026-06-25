"""Pretty terminal output using the 'rich' library."""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint

console = Console()


def print_welcome():
    console.print(Panel.fit(
        "[bold blue]AgentMind[/bold blue] — Agentic AI Research Agent\n"
        "[dim]Powered by LangGraph + Llama 3.1 (Groq) + DuckDuckGo[/dim]",
        border_style="blue"
    ))


def print_final_answer(answer: str, query: str):
    console.print("\n")
    console.print(Panel(
        Markdown(answer),
        title=f"[bold green]Research Complete[/bold green]",
        subtitle=f"[dim]Query: {query[:60]}...[/dim]" if len(query) > 60 else f"[dim]Query: {query}[/dim]",
        border_style="green",
        padding=(1, 2),
    ))


def print_agent_step(step_num: int, description: str):
    console.print(f"[bold cyan]Step {step_num}:[/bold cyan] {description}")


def print_concepts_table():
    """Print a summary of key LangGraph concepts as a table."""
    table = Table(title="LangGraph Key Concepts", border_style="blue")
    table.add_column("Concept", style="bold cyan", width=20)
    table.add_column("What it is", width=35)
    table.add_column("In this project", width=25)
    
    table.add_row(
        "StateGraph",
        "The graph container — defines the agent's structure",
        "AgentState TypedDict"
    )
    table.add_row(
        "Node",
        "A Python function that reads/updates state",
        "agent_node, tools_node"
    )
    table.add_row(
        "Edge",
        "Connection from one node to the next",
        "tools → agent (loop)"
    )
    table.add_row(
        "Conditional Edge",
        "Routes to different nodes based on logic",
        "route_after_agent()"
    )
    table.add_row(
        "Tool Binding",
        "Gives the LLM access to external tools",
        "llm.bind_tools(tools)"
    )
    table.add_row(
        "add_messages",
        "Reducer that appends instead of overwrites",
        "Annotated[list, add_messages]"
    )
    
    console.print(table)
