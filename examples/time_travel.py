"""
==================================================================
             LEARNING LAB: STATE PERSISTENCE & TIME TRAVEL        
==================================================================

In this lab, you will learn how to:
1. Persist graph state in memory using Checkpointers (MemorySaver).
2. Query and inspect the complete historical timeline of checkpoints.
3. Rewind time and fork the agent's memory to run along a new path.

Run this script:
    python examples/time_travel.py
"""

import sys
import os
# Insert parent directory so we can run directly and import tools & utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import calculator tool from the workspace
from tools.calculator import create_calculator_tool
from src.state import AgentState
from src.nodes import agent_node, tools_node, route_after_agent
from utils.display import console

# Load environment variables
load_dotenv()


# =====================================================================
# 1. BUILD GRAPH WITH MEMORY PERSISTENCE
# =====================================================================

def build_persistent_agent(checkpointer):
    # Setup LLM & Tools
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment. Please add it to your .env file.")
        
    tools_list = [create_calculator_tool()]
    tools_dict = {t.name: t for t in tools_list}
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools_list)
    
    # Initialize the StateGraph with our main AgentState
    graph = StateGraph(AgentState)
    
    # Bind nodes (using dependency injection via partial)
    from functools import partial
    graph.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools))
    graph.add_node("tools", partial(tools_node, tools_by_name=tools_dict))
    
    # Define connectivity
    graph.set_entry_point("agent")
    
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "use_tools": "tools",
            "end": END,
        }
    )
    graph.add_edge("tools", "agent")
    
    # Compile the graph with persistence Saver
    return graph.compile(checkpointer=checkpointer)


# =====================================================================
# 2. RUNNER DEMO
# =====================================================================

def main():
    console.print("[bold blue]=== LangGraph Learning Lab: State Persistence & Time Travel ===[/bold blue]")
    
    # Initialize MemorySaver
    checkpointer = MemorySaver()
    
    # Compile graph with memory checkpointer
    app = build_persistent_agent(checkpointer)
    
    # 1. RUN THE INITIAL THREAD
    # =================================================================
    thread_id = "learning-thread-001"
    config = {"configurable": {"thread_id": thread_id}}
    
    query = "Calculate 55 * 88 and then add 100."
    from src.state import create_initial_state
    initial_state = create_initial_state(query)
    
    console.print(f"\n[bold green]--- Starting Timeline 1 ---[/bold green]")
    console.print(f"[bold]Query:[/bold] {query}")
    console.print("[dim]Invoking agent...[/dim]")
    
    # Stream the nodes execution
    for event in app.stream(initial_state, config):
        node_name = list(event.keys())[0]
        console.print(f"  [yellow]Ran Node:[/yellow] {node_name}")
    # Get final state at the end of Timeline 1
    final_state = app.get_state(config)
    console.print("\n[bold green]Timeline 1 Messages:[/bold green]")
    for m in final_state.values["messages"]:
        console.print(f"  {type(m).__name__}: {m.content}")
    
    # 2. INSPECT STATE HISTORY
    # =================================================================
    console.print("\n[bold green]--- Inspecting State History ---[/bold green]")
    
    # Fetch checkpoints ordered from newest to oldest. We reverse it for chronological display.
    history = list(app.get_state_history(config))
    
    # We will keep track of the checkpoint right after the agent's first thought,
    # which is the second checkpoint in chronological order (Index 1).
    # Index 0: START node setup (System message + user query)
    # Index 1: Node 'agent' finished executing (produced tool calls to calculator)
    # Index 2: Node 'tools' finished executing (evaluated math result)
    # Index 3: Node 'agent' finished executing (compiled final response)
    
    first_agent_checkpoint = None
    
    # Reverse history to trace forward
    history_chronological = list(reversed(history))
    
    for i, checkpoint in enumerate(history_chronological):
        ckpt_id = checkpoint.config["configurable"]["checkpoint_id"]
        next_node = checkpoint.next
        messages = checkpoint.values.get("messages", [])
        console.print(f"[{i}] Checkpoint ID: [cyan]{ckpt_id}[/cyan]")
        console.print(f"    Next scheduled node: [yellow]{next_node}[/yellow]")
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                console.print(f"    Last Message: AIMessage (Tool Calls: {last_msg.tool_calls})")
            else:
                content_preview = last_msg.content[:80] + "..." if len(last_msg.content) > 80 else last_msg.content
                console.print(f"    Last Message: {type(last_msg).__name__} (Content: '{content_preview}')")
        else:
            console.print("    Last Message: None (State empty)")
        
        # Capture the checkpoint after START node setup but before agent node runs
        if checkpoint.next == ("agent",) and len(checkpoint.values.get("messages", [])) > 0:
            first_agent_checkpoint = checkpoint
            break
            
    # 3. REWIND AND FORK STATE (TIME TRAVEL)
    # =================================================================
    if first_agent_checkpoint is None:
        console.print("[red]Error: Could not locate checkpoint before agent execution.[/red]")
        return

    console.print(f"\n[bold green]--- Rewinding & Forking History (Time Travel) ---[/bold green]")
    
    past_ckpt_id = first_agent_checkpoint.config["configurable"]["checkpoint_id"]
    console.print(f"Rewinding to checkpoint index (Checkpoint ID: [cyan]{past_ckpt_id}[/cyan])")
    
    # Use the config from the historical checkpoint directly
    fork_config = first_agent_checkpoint.config
    
    # Let's inspect the historical state at that point
    past_state = app.get_state(fork_config)
    console.print(f"Historical next node was: [yellow]{past_state.next}[/yellow]")
    
    # Forking: We replace the initial human message with a new query.
    # To overwrite a message in a message list using the add_messages reducer,
    # we must supply a message with the EXACT same ID as the message we wish to replace.
    last_msg = past_state.values["messages"][-1]
    new_message = HumanMessage(content="Calculate 2 + 2.", id=last_msg.id)
    
    console.print(f"[dim]Replacing HumanMessage ID '{last_msg.id}' with new instruction...[/dim]")
    
    fork_config = app.update_state(
        fork_config,
        {"messages": [new_message]}
    )
    
    # Observe: after update_state, the graph writes a new checkpoint on the fork.
    # Let's run app.stream passing `None` as the input. It will resume from the newly written checkpoint on fork_config.
    console.print("[dim]Resuming agent on the branched history...[/dim]")
    
    for event in app.stream(None, fork_config):
        node_name = list(event.keys())[0]
        console.print(f"  [yellow]Ran Node (Fork):[/yellow] {node_name}")
        
    # Get final state at the end of the fork (using thread_id only, which fetches the latest active checkpoint)
    fork_final_state = app.get_state({"configurable": {"thread_id": thread_id}})
    console.print("\n[bold green]Branched Timeline Messages:[/bold green]")
    for m in fork_final_state.values["messages"]:
        console.print(f"  {type(m).__name__}: {m.content}")


if __name__ == "__main__":
    main()
