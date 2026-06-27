"""
==================================================================
             LEARNING LAB: REAL-TIME EVENT STREAMING              
==================================================================

In this lab, you will learn how to:
1. Initialize an asynchronous run loop using Python's asyncio.
2. Consume the modern astream_events (version 2) API.
3. Output the LLM's text token-by-token as it generates.
4. Capture node transitions and tool start/end audit logs.

Run this script:
    python examples/streaming_lab.py
"""

import sys
import os
import asyncio
# Insert parent directory so we can run directly and import tools & utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import calculator tool from the workspace
from tools.calculator import create_calculator_tool
from src.state import AgentState, create_initial_state
from src.nodes import agent_node, tools_node, route_after_agent
from utils.display import console

# Load environment variables
load_dotenv()


# =====================================================================
# 1. BUILD GRAPH
# =====================================================================

def build_agent(checkpointer):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment.")
        
    tools_list = [create_calculator_tool()]
    tools_dict = {t.name: t for t in tools_list}
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools_list)
    
    graph = StateGraph(AgentState)
    
    from functools import partial
    graph.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools))
    graph.add_node("tools", partial(tools_node, tools_by_name=tools_dict))
    
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
    
    return graph.compile(checkpointer=checkpointer)


# =====================================================================
# 2. ASYNC STREAMING CONSUMER
# =====================================================================

async def run_streaming_demo(app, initial_state, config):
    console.print("\n=== Streaming Graph Events in Real-Time ===")
    
    # app.astream_events is an async generator yielding execution events
    async for event in app.astream_events(initial_state, config, version="v2"):
        kind = event["event"]
        
        # A. Detect when a graph node starts executing
        if kind == "on_chain_start" and "node" in event["metadata"]:
            node_name = event["metadata"]["node"]
            console.print(f"\n\n[bold purple]>>> Entering Node: {node_name}[/bold purple]")
            
        # B. Detect when a chat model (LLM) streams a new token chunk
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # Extract raw string content from chunk (tool calls don't have text content)
            if chunk.content:
                print(chunk.content, end="", flush=True)
                
        # C. Detect when a tool execution starts
        elif kind == "on_tool_start":
            tool_name = event["name"]
            tool_input = event["data"].get("input", {})
            console.print(f"\n[bold cyan]🔧 Running Tool '{tool_name}' with args: {tool_input}[/bold cyan]")
            
        # D. Detect when a tool execution completes
        elif kind == "on_tool_end":
            tool_output = event["data"].get("output", "")
            console.print(f"\n[bold green]✓ Tool Output: {tool_output}[/bold green]")


# =====================================================================
# 3. ASYNC MAIN RUNNER
# =====================================================================

async def main():
    console.print("[bold blue]=== LangGraph Learning Lab: Async Event Streaming ===[/bold blue]")
    
    # Memory checkpointer is required to use astream_events/stream in LangGraph
    checkpointer = MemorySaver()
    app = build_agent(checkpointer)
    
    query = "Calculate 15 * 15 and then add 50."
    config = {"configurable": {"thread_id": "streaming-thread-001"}}
    initial_state = create_initial_state(query)
    
    console.print(f"[bold]Query:[/bold] {query}")
    console.print("[dim]Starting async run...[/dim]")
    
    # Run the streaming demo
    await run_streaming_demo(app, initial_state, config)
    
    # Retrieve final state at the end
    final_state = app.get_state(config)
    console.print("\n\n[bold green]=== Graph Completed ===[/bold green]")
    console.print("[bold]Final Answer:[/bold]")
    console.print(final_state.values["messages"][-1].content)


if __name__ == "__main__":
    # Start the asyncio event loop
    asyncio.run(main())
