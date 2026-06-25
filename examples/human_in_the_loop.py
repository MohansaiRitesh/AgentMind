"""
╔══════════════════════════════════════════════════════════════════╗
║           ADVANCED CONCEPT: HUMAN-IN-THE-LOOP (HITL)             ║
║                                                                  ║
║  One of LangGraph's most powerful features: you can PAUSE        ║
║  the agent mid-execution and wait for human input.               ║
║                                                                  ║
║  Use cases:                                                      ║
║  - Approve before sending an email                               ║
║  - Review a plan before executing expensive API calls            ║
║  - Validate AI-generated content before publishing               ║
║  - Safety checkpoints in autonomous agents                       ║
║                                                                  ║
║  HOW IT WORKS:                                                   ║
║  1. Add interrupt_before=["node_name"] to compile()              ║
║  2. Agent pauses BEFORE that node runs                           ║
║  3. State is saved to checkpointer (MemorySaver/DB)              ║
║  4. Human reviews, approves, or modifies                         ║
║  5. Resume with app.invoke(None, config) to continue             ║
╚══════════════════════════════════════════════════════════════════╝

Run this file directly:
    python examples/human_in_the_loop.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from functools import partial
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.nodes import agent_node, tools_node, route_after_agent
from tools.search import create_search_tool
from tools.calculator import create_calculator_tool, create_summarizer_tool
from utils.display import console


def build_hitl_agent():
    """
    Builds an agent with Human-in-the-Loop checkpointing.
    
    KEY DIFFERENCE from regular agent:
    1. We add a MemorySaver checkpointer — stores state to memory
       (in production, use SqliteSaver or PostgresSaver for persistence)
    2. We pass interrupt_before=["tools"] to compile() — this tells
       LangGraph to PAUSE just before the tools node runs
    3. Every run needs a thread_id config — this identifies the
       "conversation thread" for state persistence
    """
    tools_list = [create_search_tool(), create_calculator_tool(), create_summarizer_tool()]
    tools_dict = {t.name: t for t in tools_list}
    
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools_list)
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools))
    graph.add_node("tools", partial(tools_node, tools_by_name=tools_dict))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"use_tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    
    # ── THE HITL MAGIC ────────────────────────────────────────────────
    # MemorySaver stores agent state in a Python dict in memory.
    # For production use: langgraph.checkpoint.sqlite.SqliteSaver
    checkpointer = MemorySaver()
    
    # interrupt_before=["tools"] means: PAUSE the graph just before
    # the "tools" node executes. The agent has already decided WHAT
    # tool to call, but hasn't called it yet — perfect review point!
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],  # Pause here for human review
    )
    
    return app


def run_hitl_demo():
    """
    Demonstrates a full human-in-the-loop cycle.
    
    Flow:
    1. User submits query
    2. Agent thinks and plans tool calls
    3. ← PAUSE: human sees the planned tool calls
    4. Human approves or skips
    5. Agent continues executing tools and producing answer
    """
    console.print("\n[bold blue]━━━ Human-in-the-Loop Demo ━━━[/bold blue]")
    console.print("[dim]The agent will pause before each tool call for your approval[/dim]\n")
    
    app = build_hitl_agent()
    
    query = "What is the population of India and what percentage is that of the world?"
    console.print(f"[bold]Query:[/bold] {query}\n")
    
    # ── THREAD CONFIG ─────────────────────────────────────────────────
    # Every invocation needs a config with a thread_id.
    # This is how LangGraph identifies which "conversation" to
    # save/restore state for. Same thread_id = same conversation.
    config = {"configurable": {"thread_id": "hitl-demo-001"}}
    
    from src.state import create_initial_state
    initial_state = create_initial_state(query)
    
    step = 0
    # ── FIRST INVOCATION ──────────────────────────────────────────────
    # The agent will run until it hits the interrupt point (before tools)
    # Then app.invoke() returns with the partial state.
    console.print("[cyan]Starting agent...[/cyan]")
    
    for event in app.stream(initial_state, config):
        node_name = list(event.keys())[0]
        if node_name == "agent":
            console.print(f"[purple]✓ Agent node ran[/purple]")
    
    # ── CHECK STATE AFTER INTERRUPT ───────────────────────────────────
    # app.get_state(config) retrieves the current saved state
    current_state = app.get_state(config)
    
    # next tells us where the graph is paused
    console.print(f"\n[yellow]⏸ Graph paused. Next node: {current_state.next}[/yellow]")
    
    # Show what tool calls the agent wants to make
    last_msg = current_state.values["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        console.print("\n[bold]Agent wants to call these tools:[/bold]")
        for tc in last_msg.tool_calls:
            console.print(f"  [cyan]→ {tc['name']}[/cyan] with args: {tc['args']}")
    
    # ── HUMAN REVIEW POINT ────────────────────────────────────────────
    console.print("\n[bold yellow]HUMAN REVIEW:[/bold yellow]")
    approval = input("  Approve these tool calls? (y/n): ").strip().lower()
    
    if approval != "y":
        console.print("[red]Tool calls rejected. Stopping.[/red]")
        return
    
    # ── RESUME EXECUTION ──────────────────────────────────────────────
    # Pass None as input (state is already saved in checkpointer)
    # Pass the same config to identify the thread
    # The graph resumes from where it was paused!
    console.print("\n[green]Resuming agent execution...[/green]\n")
    
    final_state = None
    for event in app.stream(None, config):
        node_name = list(event.keys())[0]
        console.print(f"[dim]Ran node: {node_name}[/dim]")
        final_state = event[node_name]
    
    # Get the complete final state
    final = app.get_state(config)
    messages = final.values["messages"]
    
    # Find the last text-only AI message
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                console.print(f"\n[bold green]Final Answer:[/bold green]")
                console.print(msg.content)
                break


# ─────────────────────────────────────────────────────────────────────
# ADVANCED CONCEPT 2: MULTI-AGENT SYSTEMS
#
# LangGraph also supports multiple agents working together.
# Each agent is a subgraph, and they communicate via state.
#
# Common patterns:
# - Supervisor: one LLM routes tasks to specialist agents
# - Parallel: multiple agents run simultaneously, results merged
# - Sequential: agent A's output feeds agent B
#
# Example supervisor pseudocode:
#
# def supervisor_node(state):
#     # Supervisor LLM decides which specialist to call
#     decision = supervisor_llm.invoke(state["messages"])
#     return {"next_agent": decision.content}
#
# def route_to_specialist(state):
#     return state["next_agent"]  # "researcher", "writer", "coder"
#
# graph.add_conditional_edges("supervisor", route_to_specialist, {
#     "researcher": "research_agent",
#     "writer": "writing_agent",
#     "coder": "code_agent",
#     "FINISH": END,
# })
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    run_hitl_demo()
