import re
from functools import partial
from typing import Dict
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.errors import NodeInterrupt
from src.state import AgentState
from utils.display import console


# ─────────────────────────────────────────────────────────────────────
# NODE 1: RESEARCHER NODE (Hierarchical child subgraph execution)
# ─────────────────────────────────────────────────────────────────────

def researcher_node(state: AgentState, compiled_subgraph) -> dict:
    """
    Executes the nested researcher child subgraph.
    """
    topic = state.get("original_query", "")
    if not topic and state["messages"]:
        # Fallback to last message if original_query is missing
        topic = state["messages"][-1].content

    console.print(f"\n[bold purple]🔎 Starting researcher subgraph for topic: '{topic}'...[/bold purple]")
    
    # Run the child subgraph synchronously
    subgraph_output = compiled_subgraph.invoke({"topic": topic})
    summary = subgraph_output.get("final_summary", "No summary found.")
    
    console.print("[green]✓ Researcher subgraph execution finished.[/green]")
    
    return {
        "research_findings": [f"Topic: {topic}\nSummary of findings:\n{summary}"],
        "execution_logs": [f"Executed researcher subgraph for topic: '{topic}'."]
    }


# ─────────────────────────────────────────────────────────────────────
# NODE 2: AGENT NODE (With token tracking)
# ─────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState, llm_with_tools) -> dict:
    """
    Reasoning agent node that invokes the LLM and tracks token counts.
    """
    console.print(f"\n[bold purple]🧠 Agent thinking...[/bold purple]")
    
    # Force conclusion if tool call limit is reached
    if state.get("tool_call_count", 0) >= 8:
        console.print("[yellow]⚠ Max tool calls reached, forcing conclusion[/yellow]")
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
        # Construct message list: Inject research findings context right after system prompt
        messages = []
        if len(state["messages"]) > 0 and isinstance(state["messages"][0], SystemMessage):
            messages.append(state["messages"][0])
            history_start = 1
        else:
            history_start = 0
            
        if state.get("research_findings"):
            findings_str = "\n\n".join(state["research_findings"])
            messages.append(SystemMessage(content=(
                f"Background context gathered from research phase:\n"
                f"=== RESEARCH FINDINGS ===\n"
                f"{findings_str}\n"
                f"=========================\n"
                f"Use the findings above to help answer the user's query."
            )))
            
        messages.extend(state["messages"][history_start:])
        response = llm_with_tools.invoke(messages)
    
    # Extract token usage metadata
    metadata = response.response_metadata.get("token_usage", {})
    p_tokens = metadata.get("prompt_tokens", 0)
    c_tokens = metadata.get("completion_tokens", 0)
    t_tokens = metadata.get("total_tokens", 0)
    
    # Log details
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            console.print(f"  [cyan]→ Calling tool:[/cyan] [bold]{tc['name']}[/bold]")
            console.print(f"    [dim]Args: {tc['args']}[/dim]")
        log_entry = f"Agent invoked tool calls. Tokens used: prompt={p_tokens}, completion={c_tokens}, total={t_tokens}"
    else:
        console.print(f"  [green]→ Producing final answer[/green]")
        log_entry = f"Agent generated final response. Tokens used: prompt={p_tokens}, completion={c_tokens}, total={t_tokens}"
        
    return {
        "messages": [response],
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "total_tokens": t_tokens,
        "execution_logs": [log_entry]
    }


# ─────────────────────────────────────────────────────────────────────
# NODE 3: VALIDATOR NODE (With NodeInterrupt safety check)
# ─────────────────────────────────────────────────────────────────────

def validator_node(state: AgentState) -> dict:
    """
    Dedicated validation node that intercepts tool calls and throws interrupts
    if they exceed safety limits (e.g. calculator tool values > 1000).
    """
    console.print("[bold yellow]🛡️ Checking safety gates...[/bold yellow]")
    
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    last_msg = messages[-1]
    
    # Check if the last message is an AI message with tool calls
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            # Rule: calculator expressions shouldn't have values > 1000
            if tc["name"] == "calculator":
                expr = tc["args"].get("expression", "")
                numbers = [float(n) for n in re.findall(r'\d+\.?\d*', expr)]
                
                if any(n > 1000 for n in numbers):
                    # Check approval
                    if not state.get("is_approved", False):
                        console.print(f"  [red]CRITICAL: Safety gate triggered. Large values in calculator expression: '{expr}'[/red]")
                        raise NodeInterrupt(
                            f"Safety Check Required: Calculator expression '{expr}' contains values > 1000."
                        )
                    else:
                        console.print("  [green]Safety rule bypassed via manual human approval.[/green]")
                else:
                    console.print(f"  [green]Safety check passed (Expression: {expr})[/green]")
                    
    return {
        "execution_logs": ["Safety validation node executed successfully."]
    }


# ─────────────────────────────────────────────────────────────────────
# NODE 4: TOOLS NODE
# ─────────────────────────────────────────────────────────────────────

def tools_node(state: AgentState, tools_by_name: dict) -> dict:
    """
    Executes tool calls requested by the agent.
    """
    last_message = state["messages"][-1]
    tool_messages = []
    log_entries = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        console.print(f"\n[bold amber]🔧 Running tool:[/bold amber] [cyan]{tool_name}[/cyan] with args: {tool_args}")
        
        if tool_name in tools_by_name:
            tool = tools_by_name[tool_name]
            try:
                result = tool.invoke(tool_args)
                result_str = str(result)
                console.print(f"  [green]✓ Result ({len(result_str)} chars)[/green]")
                log_entries.append(f"Tool '{tool_name}' executed. Result: {result_str[:200]}...")
            except Exception as e:
                result_str = f"Tool error: {str(e)}"
                console.print(f"  [red]✗ Error: {e}[/red]")
                log_entries.append(f"Tool '{tool_name}' failed with error: {e}")
        else:
            result_str = f"Unknown tool: {tool_name}"
            log_entries.append(f"Tool '{tool_name}' not found.")
            
        tool_messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_call_id,
            )
        )
        
    return {
        "messages": tool_messages,
        "execution_logs": log_entries
    }


# ─────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────

def route_after_validator(state: AgentState) -> str:
    """
    Decides routing destination based on the last AI message.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        console.print(f"  [dim]→ Routing to: tools_node[/dim]")
        return "use_tools"
    else:
        console.print(f"  [dim]→ Routing to: END[/dim]")
        return "end"

# Backwards compatibility alias for legacy examples
route_after_agent = route_after_validator
