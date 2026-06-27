"""
==================================================================
             LEARNING LAB: INTERACTIVE APPROVAL & HITL            
==================================================================

In this lab, you will learn how to:
1. Raise dynamic interrupts (NodeInterrupt) from inside a node.
2. Edit tool arguments while the graph is paused.
3. Skip a node entirely by writing a mock result as_node.
4. Set up state-level approval gates to resume paused runs.

Run this script:
    python examples/interactive_approval.py
"""

import sys
import os
import re
# Insert parent directory so we can run directly and import tools & utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt

# Import calculator tool from the workspace
from tools.calculator import create_calculator_tool
from utils.display import console

# Load environment variables
load_dotenv()


# =====================================================================
# 1. STATE DEFINITION
# =====================================================================

class ApprovalState(TypedDict):
    """ State class mapping messages and approval metadata. """
    messages: Annotated[list, add_messages]
    is_approved: bool  # Authorization gate set by humans during pause


# =====================================================================
# 2. NODE DEFINITIONS
# =====================================================================

def agent_node(state: ApprovalState, llm_with_tools) -> dict:
    """ Core LLM node deciding whether to call calculator. """
    console.print("\n[bold purple][Node: agent] Invoking LLM...[/bold purple]")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def validator_node(state: ApprovalState) -> dict:
    """
    Dedicated validation node that intercepts tool calls and raises interrupts.
    
    Why use a separate node instead of raising inside agent_node?
    - If we raise NodeInterrupt inside agent_node, the AIMessage with tool calls
      is NEVER written to the state.
    - By having agent_node run and return the message, the checkpointer saves it.
    - The validator_node then reads the saved message and raises NodeInterrupt.
    - When we resume, validator_node executes again. It checks state['is_approved'].
      If True, it lets it pass through.
    """
    console.print("[bold yellow][Node: validator] Checking safety rules...[/bold yellow]")
    
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    last_msg = messages[-1]
    
    # If the last message is an AI tool call, check if calculator uses large numbers
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            if tc["name"] == "calculator":
                expr = tc["args"].get("expression", "")
                
                # Heuristic: find all numbers in the expression
                numbers = [float(n) for n in re.findall(r'\d+\.?\d*', expr)]
                
                # Safety Gate Rule: trigger review if any number exceeds 1000
                if any(n > 1000 for n in numbers):
                    # Check if already approved
                    if not state.get("is_approved", False):
                        console.print(f"  [red]CRITICAL: Safety violation detected. Expression contains numbers > 1000: {expr}[/red]")
                        raise NodeInterrupt(
                            f"Review Required: Calculator expression '{expr}' contains values > 1000."
                        )
                    else:
                        console.print("  [green]Safety rule bypassed via manual human approval flag.[/green]")
                else:
                    console.print(f"  [green]Safety check passed (Expression: {expr}).[/green]")
                    
    return {}


def tools_node(state: ApprovalState, tools_by_name: dict) -> dict:
    """ Node that executes the calculator tool. """
    console.print("\n[bold cyan][Node: tools] Executing tool calls...[/bold cyan]")
    last_message = state["messages"][-1]
    tool_messages = []
    
    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc["id"]
        
        console.print(f"  Running: {tool_name} with args: {tool_args}")
        tool = tools_by_name[tool_name]
        result = tool.invoke(tool_args)
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
        
    return {"messages": tool_messages}


# =====================================================================
# 3. ROUTING AND BUILD GRAPH
# =====================================================================

def route_after_validator(state: ApprovalState) -> str:
    """ Decides whether to route to tools or to END. """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tools"
    return "end"


def build_hitl_agent(checkpointer):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment.")
        
    tools_list = [create_calculator_tool()]
    tools_dict = {t.name: t for t in tools_list}
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools_list)
    
    # Initialize state graph
    graph = StateGraph(ApprovalState)
    
    # Bind nodes
    from functools import partial
    graph.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools))
    graph.add_node("validator", validator_node)
    graph.add_node("tools", partial(tools_node, tools_by_name=tools_dict))
    
    # Define execution paths
    graph.set_entry_point("agent")
    graph.add_edge("agent", "validator")
    
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "use_tools": "tools",
            "end": END,
        }
    )
    graph.add_edge("tools", "agent")
    
    # Compile with memory saver checkpointer
    return graph.compile(checkpointer=checkpointer)


# =====================================================================
# 4. RUNNER DEMO (TESTING ALL 3 WORKFLOWS)
# =====================================================================

def main():
    console.print("[bold blue]=== LangGraph Learning Lab: Interactive HITL & Approvals ===[/bold blue]")
    
    # Setup agent query requiring approval
    query = "Calculate 5000 * 2."
    system_prompt = (
        "You are a calculation agent. You have access to the 'calculator' tool. "
        "Use the calculator for any math operations. Speak only through the calculator tool initially."
    )
    
    # =================================================================
    # WORKFLOW 1: MUTATING/EDITING THE STATE DURING PAUSE
    # =================================================================
    console.print("\n[bold green]------ WORKFLOW 1: EDITING STATE ------[/bold green]")
    checkpointer = MemorySaver()
    app = build_hitl_agent(checkpointer)
    
    config = {"configurable": {"thread_id": "thread-edit-test"}}
    initial_state = {
        "messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)],
        "is_approved": False,
    }
    
    console.print(f"Query: {query}")
    console.print("[dim]Starting execution. Expecting interrupt...[/dim]")
    
    # Run the graph and catch the dynamic interrupt exception
    try:
        app.invoke(initial_state, config)
    except Exception as e:
        # Check if it was a NodeInterrupt
        if "NodeInterrupt" in type(e).__name__ or isinstance(e, NodeInterrupt):
            console.print(f"\n[bold red]⏸️ Interrupted: {e}[/bold red]")
            
            # Retrieve the paused state
            paused_state = app.get_state(config)
            last_message = paused_state.values["messages"][-1]
            old_expression = last_message.tool_calls[0]["args"]["expression"]
            console.print(f"Tool request was: calculator('{old_expression}')")
            
            # MUTATE STATE: Edit the arguments to make it safe (< 1000)
            new_expression = "500 * 2"
            console.print(f"MUTATING arguments to safe value: calculator('{new_expression}')")
            last_message.tool_calls[0]["args"]["expression"] = new_expression
            
            # Update state with modified messages. This returns the new checkpoint config.
            new_config = app.update_state(config, {"messages": [last_message]})
            
            # Resume run by passing None as input and the new config
            console.print("[dim]Resuming execution...[/dim]")
            final_state = app.invoke(None, new_config)
            
            console.print(f"\n[bold green]Final Response after EDIT:[/bold green]")
            console.print(f"  {final_state['messages'][-1].content}")
        else:
            raise e

    # =================================================================
    # WORKFLOW 2: SKIPPING THE TOOL NODE (MOCKING TOOL OUTPUT)
    # =================================================================
    console.print("\n[bold green]------ WORKFLOW 2: MOCKING & SKIPPING TOOLS ------[/bold green]")
    checkpointer = MemorySaver()
    app = build_hitl_agent(checkpointer)
    
    config = {"configurable": {"thread_id": "thread-skip-test"}}
    initial_state = {
        "messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)],
        "is_approved": False,
    }
    
    console.print(f"Query: {query}")
    console.print("[dim]Starting execution. Expecting interrupt...[/dim]")
    
    try:
        app.invoke(initial_state, config)
    except Exception as e:
        if "NodeInterrupt" in type(e).__name__ or isinstance(e, NodeInterrupt):
            console.print(f"\n[bold red]⏸️ Interrupted: {e}[/bold red]")
            
            paused_state = app.get_state(config)
            last_message = paused_state.values["messages"][-1]
            tool_call_id = last_message.tool_calls[0]["id"]
            
            # MOCK OUTPUT: Write a custom tool response directly
            mock_content = "Mocked calculation result: 42"
            console.print(f"MOCKING output to skip tools node: '{mock_content}'")
            mock_tool_msg = ToolMessage(content=mock_content, tool_call_id=tool_call_id)
            
            # Update state as if the 'tools' node wrote it!
            # The as_node="tools" tells LangGraph that 'tools' node has already run.
            new_config = app.update_state(
                config, 
                {"messages": [mock_tool_msg]}, 
                as_node="tools"
            )
            
            # Resume execution (will bypass tools node and route directly back to agent)
            console.print("[dim]Resuming execution...[/dim]")
            final_state = app.invoke(None, new_config)
            
            console.print(f"\n[bold green]Final Response after SKIP & MOCK:[/bold green]")
            console.print(f"  {final_state['messages'][-1].content}")
        else:
            raise e

    # =================================================================
    # WORKFLOW 3: HUMAN APPROVAL GATE
    # =================================================================
    console.print("\n[bold green]------ WORKFLOW 3: APPROVAL GATE ------[/bold green]")
    checkpointer = MemorySaver()
    app = build_hitl_agent(checkpointer)
    
    config = {"configurable": {"thread_id": "thread-approval-test"}}
    initial_state = {
        "messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)],
        "is_approved": False,
    }
    
    console.print(f"Query: {query}")
    console.print("[dim]Starting execution. Expecting interrupt...[/dim]")
    
    try:
        app.invoke(initial_state, config)
    except Exception as e:
        if "NodeInterrupt" in type(e).__name__ or isinstance(e, NodeInterrupt):
            console.print(f"\n[bold red]⏸️ Interrupted: {e}[/bold red]")
            
            # APPROVE: Set is_approved flag to True
            console.print("APPROVING execution as-is (setting is_approved=True)")
            new_config = app.update_state(config, {"is_approved": True})
            
            # Resume execution (will let tools node execute with original inputs)
            console.print("[dim]Resuming execution...[/dim]")
            final_state = app.invoke(None, new_config)
            
            console.print(f"\n[bold green]Final Response after APPROVAL:[/bold green]")
            console.print(f"  {final_state['messages'][-1].content}")
        else:
            raise e


if __name__ == "__main__":
    main()
