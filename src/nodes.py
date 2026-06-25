"""
╔══════════════════════════════════════════════════════════════════╗
║                    CONCEPT: NODES                                ║
║                                                                  ║
║  A NODE is just a Python function with this signature:           ║
║                                                                  ║
║    def my_node(state: AgentState) -> dict:                       ║
║        # do something with state                                 ║
║        return {"key": new_value}  # partial state update         ║
║                                                                  ║
║  LangGraph automatically:                                        ║
║    1. Passes the full current state to the node                  ║
║    2. Takes the returned dict and MERGES it into state           ║
║    3. Moves to the next node (based on edges)                    ║
║                                                                  ║
║  You DON'T return the full state — just the parts you changed.   ║
╚══════════════════════════════════════════════════════════════════╝
"""

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from src.state import AgentState
from utils.display import console
import json


# ─────────────────────────────────────────────────────────────────────
# NODE 1: AGENT NODE
# This is the "brain" — calls the LLM to decide what to do next.
# The LLM can either:
#   A) Call one or more tools (the agent's "hands")
#   B) Produce a final answer (no tools = we're done)
# ─────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState, llm_with_tools) -> dict:
    """
    The core reasoning node.
    
    HOW IT WORKS:
    1. We pass the full conversation history (state["messages"]) to the LLM
    2. The LLM with bound tools can EITHER:
       - Return text only → agent is done, no more tool calls
       - Return tool_calls → agent wants to use tools
    3. We check if tool_call_count is too high → force conclusion
    
    CONCEPT: Tool Binding
    When we do `llm.bind_tools(tools)`, we're giving the LLM a
    "menu" of tools it can call. The LLM outputs a special structured
    JSON response like:
    {
      "tool_calls": [{
        "name": "web_search",
        "args": {"query": "latest AI news"}
      }]
    }
    LangGraph then routes this to the tools_node automatically.
    """
    
    console.print(f"\n[bold purple]🧠 Agent thinking...[/bold purple]")
    
    # Safety: if too many tool calls, force the agent to conclude
    if state["tool_call_count"] >= 8:
        console.print("[yellow]⚠ Max tool calls reached, forcing conclusion[/yellow]")
        # Inject a message telling the LLM to wrap up
        from langchain_core.messages import HumanMessage
        messages = state["messages"] + [
            HumanMessage(content=(
                "You have gathered enough information. "
                "Please now write your final comprehensive answer "
                "based on all the research you have done. "
                "Do NOT call any more tools."
            ))
        ]
        response = llm_with_tools.invoke(messages)
    else:
        # Normal: let the LLM decide freely
        response = llm_with_tools.invoke(state["messages"])
    
    # Debug: show what the LLM decided to do
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            console.print(f"  [cyan]→ Calling tool:[/cyan] [bold]{tc['name']}[/bold]")
            console.print(f"    [dim]Args: {tc['args']}[/dim]")
    else:
        console.print(f"  [green]→ Producing final answer[/green]")
    
    # Return only the fields we're updating
    # LangGraph's add_messages reducer will APPEND this response
    # to the existing messages list (not overwrite!)
    return {
        "messages": [response],
        "tool_call_count": state["tool_call_count"] + 1,
    }


# ─────────────────────────────────────────────────────────────────────
# NODE 2: TOOLS NODE
# Executes the tools that the LLM requested.
# 
# CONCEPT: The ReAct Loop
# After agent_node decides "I want to call tool X", this node:
# 1. Finds which tools were requested (from the last AIMessage)
# 2. Runs each tool with the provided arguments
# 3. Wraps results in ToolMessage objects
# 4. Adds them to the messages list
# Then graph goes BACK to agent_node (the loop!)
# ─────────────────────────────────────────────────────────────────────

def tools_node(state: AgentState, tools_by_name: dict) -> dict:
    """
    Executes all tool calls requested by the last agent response.
    
    CONCEPT: ToolMessage
    After a tool runs, we create a ToolMessage:
    - tool_call_id: links back to the specific tool_call in the AIMessage
    - content: the result of the tool
    
    LangChain/LangGraph requires this linking — the LLM needs to know
    WHICH tool call produced WHICH result (especially if multiple tools
    are called in parallel).
    """
    
    # Get the last message (should be AIMessage with tool_calls)
    last_message = state["messages"][-1]
    
    tool_messages = []
    new_findings = list(state["research_findings"])
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        console.print(f"\n[bold amber]🔧 Running tool:[/bold amber] [cyan]{tool_name}[/cyan]")
        
        # Look up the tool and run it
        if tool_name in tools_by_name:
            tool = tools_by_name[tool_name]
            try:
                result = tool.invoke(tool_args)
                result_str = str(result)
                
                console.print(f"  [green]✓ Result ({len(result_str)} chars)[/green]")
                
                # If this was a search, accumulate the finding
                if tool_name == "web_search":
                    query = tool_args.get("query", "")
                    new_findings.append(f"[Search: {query}]\n{result_str[:500]}")
                    
            except Exception as e:
                result_str = f"Tool error: {str(e)}"
                console.print(f"  [red]✗ Error: {e}[/red]")
        else:
            result_str = f"Unknown tool: {tool_name}"
        
        # CRITICAL: Create ToolMessage linking back to the tool_call_id
        tool_messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_call_id,
            )
        )
    
    return {
        "messages": tool_messages,  # add_messages will append these
        "research_findings": new_findings,
    }


# ─────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTION
# This is NOT a node — it's a routing function used in conditional edges.
#
# CONCEPT: Conditional Edges
# After agent_node runs, LangGraph calls this function.
# It returns a STRING that matches one of the defined edge destinations:
#   - "use_tools"  → go to tools_node (loop continues)
#   - "end"        → go to END (agent is done)
# ─────────────────────────────────────────────────────────────────────

def route_after_agent(state: AgentState) -> str:
    """
    Decides where to go after the agent_node runs.
    
    This function is the "brain's output decoder":
    - If the LLM produced tool calls → run those tools
    - If the LLM produced just text → we're done
    
    IMPORTANT: The return value must match a key in the
    conditional_edges dict you pass to add_conditional_edges()
    in the graph builder.
    """
    last_message = state["messages"][-1]
    
    # AIMessage.tool_calls is non-empty if the LLM wants to use tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        console.print(f"  [dim]→ Routing to: tools_node[/dim]")
        return "use_tools"
    else:
        console.print(f"  [dim]→ Routing to: END[/dim]")
        return "end"
