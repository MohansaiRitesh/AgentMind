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
import re
import uuid
import sys
import os

from utils.display import console, print_welcome, print_final_answer, print_concepts_table
from src.agent import create_agent
from src.state import create_initial_state
from langgraph.errors import NodeInterrupt
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import ToolMessage, AIMessage


# DEMO QUERIES to showcase the agent's capabilities
DEMO_QUERIES = [
    "What are the most significant AI breakthroughs in 2024, and how do they impact everyday users?",
    "Calculate 5000 * 2 to check safety validation and interrupts.",
]


def run_agent_with_interrupts(query: str, thread_id: str = None) -> str:
    """
    Runs the agent using an interactive loop that intercepts safety validator interrupts
    and displays a Human-in-the-Loop approval/editing/mocking dashboard.
    """
    if not thread_id:
        thread_id = f"thread-{uuid.uuid4().hex[:8]}"
        
    db_path = "agent_memory.db"
    console.print(f"\n[dim]Configuring SQLite checkpointer at '{db_path}' for Thread ID: [bold]{thread_id}[/bold][/dim]")
    
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = create_agent(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        # Check if the thread already has a history
        thread_state = app.get_state(config)
        
        if thread_state.values:
            console.print(f"[green]✓ Found existing history for thread '{thread_id}'. Resuming execution...[/green]")
            run_input = None
        else:
            console.print(f"[dim]Starting new execution flow...[/dim]")
            run_input = create_initial_state(query)
            
        while True:
            try:
                # Stream the execution graph steps
                for chunk in app.stream(run_input, config):
                    node_name = list(chunk.keys())[0]
                    node_output = chunk[node_name]
                    
                    if "execution_logs" in node_output and node_output["execution_logs"]:
                        for log in node_output["execution_logs"]:
                            console.print(f"  [dim]↳ [Node: {node_name}] {log}[/dim]")
                            
                # Reached the end of execution successfully
                final_state = app.get_state(config)
                messages = final_state.values.get("messages", [])
                
                final_answer = ""
                for msg in reversed(messages):
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            final_answer = msg.content
                            break
                            
                # Output final diagnostics
                console.print("\n[bold green]=== Execution Performance & Diagnostics ===[/bold green]")
                console.print(f"  - Total Accumulated Tokens: [yellow]{final_state.values.get('total_tokens', 0)}[/yellow]")
                console.print(f"  - Total Steps Logged:       [yellow]{len(final_state.values.get('execution_logs', []))}[/yellow]")
                
                return final_answer
                
            except BaseException as e:
                # Catching NodeInterrupt
                if "NodeInterrupt" not in type(e).__name__ and not isinstance(e, NodeInterrupt):
                    raise e
                    
                console.print(f"\n[bold red]⏸️ INTERRUPT DETECTED: {e}[/bold red]")
                
                # Fetch paused state to check the tool request
                paused_state = app.get_state(config)
                last_message = paused_state.values["messages"][-1]
                
                if not (isinstance(last_message, AIMessage) and last_message.tool_calls):
                    console.print("[red]Interrupt occurred but no tool calls found in the last message.[/red]")
                    raise e
                    
                tool_calls = last_message.tool_calls
                console.print("\n[bold yellow]Pending Action Authorization Request(s):[/bold yellow]")
                for tc in tool_calls:
                    console.print(f"  - Tool name: [cyan]{tc['name']}[/cyan]")
                    console.print(f"    Arguments: [cyan]{tc['args']}[/cyan]")
                    
                console.print("\n[bold]Select approval decision action:[/bold]")
                console.print("  [1] Approve (gate authorized) - run the tool calls as requested")
                console.print("  [2] Edit arguments - modify inputs before execution")
                console.print("  [3] Skip & Mock - mock tool response directly and skip executing node")
                console.print("  [4] Terminate - exit execution thread")
                
                choice = input("\nDecision [1-4]: ").strip()
                
                if choice == "1":
                    console.print("\n[green]✓ Action approved. Resuming execution...[/green]")
                    app.update_state(config, {"is_approved": True})
                    run_input = None  # Resume
                elif choice == "2":
                    modified_calls = []
                    for tc in tool_calls:
                        console.print(f"\nEditing parameters for [bold cyan]{tc['name']}[/bold cyan]:")
                        new_args = {}
                        for arg_key, arg_val in tc["args"].items():
                            val = input(f"  Enter value for '{arg_key}' (current: {arg_val}): ").strip()
                            if val == "":
                                new_args[arg_key] = arg_val
                            else:
                                # Convert to int/float if digits, else use string
                                if re.match(r'^\d+$', val):
                                    new_args[arg_key] = int(val)
                                elif re.match(r'^\d+\.\d+$', val):
                                    new_args[arg_key] = float(val)
                                else:
                                    new_args[arg_key] = val
                        tc["args"] = new_args
                        modified_calls.append(tc)
                        
                    last_message.tool_calls = modified_calls
                    app.update_state(config, {"messages": [last_message]})
                    run_input = None  # Resume
                elif choice == "3":
                    mock_messages = []
                    for tc in tool_calls:
                        mock_val = input(f"\nProvide mock result output for tool '{tc['name']}' (args: {tc['args']}): ").strip()
                        mock_messages.append(
                            ToolMessage(content=mock_val, tool_call_id=tc["id"])
                        )
                    # Update state as if tools node returned this
                    app.update_state(config, {"messages": mock_messages}, as_node="tools")
                    run_input = None  # Resume
                else:
                    console.print("[dim]Terminated. Session state remains saved in SQLite database.[/dim]")
                    raise KeyboardInterrupt("Terminated by user choice.")


def run_demo():
    """Run the agent on a sample query to demonstrate capabilities."""
    print_welcome()
    console.print("\n[bold]Running safety interrupt demo query...[/bold]")
    
    query = DEMO_QUERIES[1]
    answer = run_agent_with_interrupts(query, thread_id="demo-math-thread")
    print_final_answer(answer, query)


def run_interactive():
    """Interactive mode — loop accepting user queries."""
    print_welcome()
    console.print("\n[dim]Type your research question, or 'quit' to exit.[/dim]")
    console.print("[dim]You can also type 'thread <id>' to select a specific SQLite thread.[/dim]\n")
    
    current_thread = f"interactive-{uuid.uuid4().hex[:6]}"
    
    while True:
        try:
            query = input(f"🔍 [{current_thread}] Research query: ").strip()
            
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
                
            # Handle thread switching Command
            if query.lower().startswith("thread "):
                parts = query.split(maxsplit=1)
                if len(parts) > 1:
                    current_thread = parts[1].strip()
                    console.print(f"[green]✓ Switched active thread to '{current_thread}'[/green]")
                continue
                
            answer = run_agent_with_interrupts(query, thread_id=current_thread)
            print_final_answer(answer, query)
            console.print("\n" + "─" * 60 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error occurred: {e}[/red]")


def run_single_query(query: str):
    """Run a single query and print the result."""
    print_welcome()
    answer = run_agent_with_interrupts(query)
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
    
    # Configure console encoding safety for box/unicode characters
    if sys.platform.startswith("win"):
        # Set output stream to utf-8 mode
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
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
