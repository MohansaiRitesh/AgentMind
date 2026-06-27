import os
from dotenv import load_dotenv
from functools import partial

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START

from src.state import AgentState, create_initial_state
from tools.search import create_search_tool
from tools.calculator import create_calculator_tool
from tools.summarizer import create_summarizer_tool
from utils.display import console


load_dotenv()


def create_tools():
    """
    Create all tools available to our agent.
    """
    search = create_search_tool()
    calculator = create_calculator_tool()
    summarizer = create_summarizer_tool()
    
    tools_list = [search, calculator, summarizer]
    tools_dict = {t.name: t for t in tools_list}
    
    return tools_list, tools_dict


def create_llm(tools_list: list):
    """
    Initialize the LLM and bind tools.
    Returns both the raw LLM and the tool-bound LLM.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ GROQ_API_KEY not found!\n"
            "Get your FREE key at: https://console.groq.com\n"
            "Then add it to your .env file"
        )
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Fast free model
        temperature=0,                  # Deterministic for reasoning
        api_key=api_key,
    )
    
    llm_with_tools = llm.bind_tools(tools_list)
    
    console.print(f"[green]✓ LLM initialized:[/green] llama-3.1-8b-instant via Groq")
    return llm, llm_with_tools


def build_agent_graph(llm, llm_with_tools, tools_dict: dict, checkpointer=None):
    """
    Builds and compiles the parent LangGraph agent.
    """
    graph = StateGraph(AgentState)
    
    # ── SUBGRAPH COMPILATION ──────────────────────────────────────────
    from src.researcher_graph import create_researcher_subgraph
    compiled_subgraph = create_researcher_subgraph(llm)
    
    # ── NODE REGISTRATION ─────────────────────────────────────────────
    from src.nodes import researcher_node, agent_node, validator_node, tools_node, route_after_validator
    
    graph.add_node("researcher", partial(researcher_node, compiled_subgraph=compiled_subgraph))
    graph.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools))
    graph.add_node("validator", validator_node)
    graph.add_node("tools", partial(tools_node, tools_by_name=tools_dict))
    
    # ── CONNECTIVITY ──────────────────────────────────────────────────
    graph.set_entry_point("researcher")
    
    graph.add_edge("researcher", "agent")
    graph.add_edge("agent", "validator")
    
    # Validation conditional edge
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "use_tools": "tools",
            "end": END,
        }
    )
    
    graph.add_edge("tools", "agent")
    
    # Compile the graph with checkpointer
    app = graph.compile(checkpointer=checkpointer)
    
    console.print("[green]✓ Agent graph compiled successfully[/green]")
    return app


def create_agent(checkpointer=None):
    """Creates and returns a fully configured agent app."""
    tools_list, tools_dict = create_tools()
    llm, llm_with_tools = create_llm(tools_list)
    
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        
    app = build_agent_graph(llm, llm_with_tools, tools_dict, checkpointer)
    return app


def run_agent(query: str, thread_id: str = "default-thread", checkpointer=None) -> str:
    """
    Run the agent on a query and return the final answer.
    """
    console.print(f"\n[bold blue]━━━ AgentMind Research Agent ━━━[/bold blue]")
    console.print(f"[bold]Query:[/bold] {query}\n")
    
    app = create_agent(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(query)
    
    final_state = None
    
    # Consume and print stream
    console.print("[dim]Streaming agent execution...[/dim]\n")
    
    for chunk in app.stream(initial_state, config):
        # Retrieve name of node that completed and its updates
        node_name = list(chunk.keys())[0]
        node_output = chunk[node_name]
        final_state = node_output
        
    # Re-fetch state at the end to get full message history
    full_state = app.get_state(config)
    messages = full_state.values.get("messages", [])
    
    # Print metrics/diagnostics summary
    console.print("\n[bold green]=== Diagnostics Summary ===[/bold green]")
    console.print(f"  - Total Running Tokens: [yellow]{full_state.values.get('total_tokens', 0)}[/yellow]")
    console.print(f"  - Execution steps:")
    for log in full_state.values.get("execution_logs", []):
         console.print(f"    - {log}")
         
    # Get the last AIMessage (the agent's final response)
    final_answer = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                final_answer = msg.content
                break
    
    return final_answer
