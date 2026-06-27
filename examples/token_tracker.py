"""
==================================================================
             LEARNING LAB: STATE REDUCERS & TOKEN TRACKER         
==================================================================

In this lab, you will learn how to:
1. Write custom reducer functions to merge state updates.
2. Track running token counts (input, output, total) across nodes.
3. Keep an execution log audit trail of all node traversals.
4. Access and display this metadata dynamically.

Run this script:
    python examples/token_tracker.py
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

# Import search & calculator tools from the workspace
from tools.search import create_search_tool
from tools.calculator import create_calculator_tool
from utils.display import console

# Load environment variables
load_dotenv()


# =====================================================================
# 1. CUSTOM REDUCER FUNCTIONS
# =====================================================================

def sum_tokens(current: int | None, new: int | None) -> int:
    """
    A custom reducer function that adds two integers.
    
    Why:
    By default, LangGraph overwrites state variables (last-write wins).
    To accumulate values like token usage, we need a reducer.
    
    How it works:
    - LangGraph passes the current value of the state field as `current` (which starts as None).
    - It passes the value returned by the active node as `new`.
    - We return the sum of the two, ensuring we handle initial None values.
    """
    current_val = current if current is not None else 0
    new_val = new if new is not None else 0
    return current_val + new_val


def append_logs(current: list[str] | None, new: list[str] | str | None) -> list[str]:
    """
    A custom reducer function to compile a running execution log.
    
    Why:
    Instead of overwriting a list, we want to append new logs to it.
    
    How it works:
    - If `new` is a single string log, we append it.
    - If `new` is a list, we combine/extend the list.
    - This gives flexibility to nodes to return either `{"execution_logs": "Log msg"}`
      or `{"execution_logs": ["Log A", "Log B"]}`.
    """
    current_logs = list(current) if current is not None else []
    if new is None:
        return current_logs
    if isinstance(new, list):
        return current_logs + new
    return current_logs + [new]


# =====================================================================
# 2. STATE DEFINITION WITH REDUCERS
# =====================================================================

# We import the standard add_messages reducer
from langgraph.graph.message import add_messages

class TokenTrackerState(TypedDict):
    """
    The State object defining all fields/channels.
    Notice the use of typing.Annotated to associate our custom reducers.
    """
    # The message history. add_messages will append or overwrite by message ID.
    messages: Annotated[list, add_messages]
    
    # Token usage metrics. The sum_tokens reducer ensures nodes can return
    # local token counts, and LangGraph will accumulate them into a running sum.
    prompt_tokens: Annotated[int, sum_tokens]
    completion_tokens: Annotated[int, sum_tokens]
    total_tokens: Annotated[int, sum_tokens]
    
    # The running audit trail logs, accumulated using append_logs.
    execution_logs: Annotated[list[str], append_logs]


# =====================================================================
# 3. NODE DEFINITIONS
# =====================================================================

def agent_node(state: TokenTrackerState, llm_with_tools) -> dict:
    """
    Node that runs the LLM, makes decisions, and extracts token usage.
    """
    console.print("\n[bold purple][Node: agent] Thinking...[/bold purple]")
    
    # Invoke the LLM with current messages
    response = llm_with_tools.invoke(state["messages"])
    
    # Extract token usage metadata from the response
    # Groq API returns this under response_metadata["token_usage"]
    metadata = response.response_metadata.get("token_usage", {})
    p_tokens = metadata.get("prompt_tokens", 0)
    c_tokens = metadata.get("completion_tokens", 0)
    t_tokens = metadata.get("total_tokens", 0)
    
    log_entry = f"Agent Node executed. Prompt tokens: {p_tokens}, Completion tokens: {c_tokens}"
    
    # Return updates.
    # Note that we only need to return the delta updates. LangGraph's reducers
    # will merge these with the current state automatically!
    return {
        "messages": [response],
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "total_tokens": t_tokens,
        "execution_logs": log_entry,
    }


def tools_node(state: TokenTrackerState, tools_by_name: dict) -> dict:
    """
    Node that executes the tools requested by the LLM.
    """
    last_message = state["messages"][-1]
    tool_messages = []
    log_entries = []
    
    # Process all tool calls in parallel/sequence
    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc["id"]
        
        console.print(f"[Node: tools] Running tool: [cyan]{tool_name}[/cyan] with args: {tool_args}")
        
        if tool_name in tools_by_name:
            tool = tools_by_name[tool_name]
            try:
                result = tool.invoke(tool_args)
                result_str = str(result)
                log_entries.append(f"Tool '{tool_name}' completed successfully.")
            except Exception as e:
                result_str = f"Error: {str(e)}"
                log_entries.append(f"Tool '{tool_name}' failed: {e}")
        else:
            result_str = f"Error: Tool '{tool_name}' not found."
            log_entries.append(f"Tool '{tool_name}' lookup failed.")
            
        tool_messages.append(ToolMessage(content=result_str, tool_call_id=tool_call_id))
        
    return {
        "messages": tool_messages,
        "execution_logs": log_entries,
    }


# =====================================================================
# 4. CONDITIONAL ROUTING LOGIC
# =====================================================================

def route_after_agent(state: TokenTrackerState) -> str:
    """
    Simple router deciding whether to route to tools or to END.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tools"
    return "end"


# =====================================================================
# 5. BUILD AND COMPILE GRAPH
# =====================================================================

def build_token_tracker_agent():
    # Setup LLM & Tools
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment. Please add it to your .env file.")
        
    # We load calculator and search
    tools_list = [create_calculator_tool(), create_search_tool()]
    tools_dict = {t.name: t for t in tools_list}
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools_list)
    
    # Initialize the StateGraph with our custom TypedDict containing reducers
    graph = StateGraph(TokenTrackerState)
    
    # Bind nodes (using partial to inject the LLM and tool dictionary dependencies)
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
    
    return graph.compile()


# =====================================================================
# 6. RUNNER DEMO
# =====================================================================

def main():
    console.print("[bold blue]=== LangGraph Learning Lab: Token Tracker & Reducers ===[/bold blue]")
    
    # Compile the graph
    app = build_token_tracker_agent()
    
    # Set the initial state
    query = "Calculate 12345 * 54321 and then find the capital of France."
    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful research agent. You have access to exactly two tools: 'calculator' and 'web_search'. You MUST NOT call any other tool names (such as 'brave_search'). Use the provided tools for math or web searches."),
            HumanMessage(content=query)
        ],
        # Notice we do not pass initial values for prompt_tokens, execution_logs, etc.
        # LangGraph initializes these empty channels and passes None to our reducers.
    }
    
    console.print(f"[bold]Query:[/bold] {query}")
    console.print("\n[dim]Invoking the graph...[/dim]")
    
    # Invoke the graph synchronously
    final_state = app.invoke(initial_state)
    
    # === SHOW THE RESULT ================================================
    console.print("\n[bold green]=== Execution Summary & Diagnostics ===[/bold green]")
    
    # Print the running audit log compiled by our custom reducer append_logs
    console.print("\n[bold]Execution Logs (via custom list reducer):[/bold]")
    for i, log in enumerate(final_state["execution_logs"], 1):
        console.print(f"  {i}. {log}")
        
    # Print accumulated token counts compiled by our custom reducer sum_tokens
    console.print("\n[bold]Total Accumulated Token Usage (via custom sum reducer):[/bold]")
    console.print(f"  - Prompt (Input) Tokens:      [yellow]{final_state['prompt_tokens']}[/yellow]")
    console.print(f"  - Completion (Output) Tokens: [yellow]{final_state['completion_tokens']}[/yellow]")
    console.print(f"  - Total Running Tokens:       [yellow]{final_state['total_tokens']}[/yellow]")
    
    # Show the final response text
    last_msg = final_state["messages"][-1]
    console.print("\n[bold green]Final Agent Response:[/bold green]")
    console.print(last_msg.content)


if __name__ == "__main__":
    main()
