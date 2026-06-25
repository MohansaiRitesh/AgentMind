"""
╔══════════════════════════════════════════════════════════════════╗
║                    AGENTMIND — MAIN ENTRY POINT                  ║
╚══════════════════════════════════════════════════════════════════╝

Run modes:
  python main.py                        → Demo with example queries
  python main.py --interactive          → Interactive prompt loop
  python main.py --query "your query"   → Single query
  python main.py --concepts             → Print concepts overview
"""

import argparse

from utils.display import console, print_welcome, print_final_answer, print_concepts_table
from src.agent import run_agent


# DEMO QUERIES to showcase the agent's capabilities
DEMO_QUERIES = [
    "What are the most significant AI breakthroughs in 2024, and how do they impact everyday users?",
    "What is quantum computing and what problems can it solve that regular computers can't?",
]


def run_demo():
    """Run the agent on a sample query to demonstrate capabilities."""
    print_welcome()
    console.print("\n[bold]Running demo query...[/bold]")
    
    query = DEMO_QUERIES[1]
    answer = run_agent(query)
    print_final_answer(answer, query)


def run_interactive():
    """Interactive mode — loop accepting user queries."""
    print_welcome()
    console.print("\n[dim]Type your research question, or 'quit' to exit[/dim]\n")
    
    while True:
        try:
            query = input("🔍 Research query: ").strip()
            
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
            
            answer = run_agent(query)
            print_final_answer(answer, query)
            console.print("\n" + "─" * 60 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[dim]Check your GROQ_API_KEY in .env[/dim]")


def run_single_query(query: str):
    """Run a single query and print the result."""
    print_welcome()
    answer = run_agent(query)
    print_final_answer(answer, query)


def main():
    parser = argparse.ArgumentParser(
        description="AgentMind — Agentic AI Research Agent powered by LangGraph"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Research query to run"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--concepts",
        action="store_true",
        help="Print key LangGraph concepts overview"
    )
    
    args = parser.parse_args()
    
    if args.concepts:
        print_concepts_table()
    elif args.interactive:
        run_interactive()
    elif args.query:
        run_single_query(args.query)
    else:
        run_demo()


if __name__ == "__main__":
    main()
