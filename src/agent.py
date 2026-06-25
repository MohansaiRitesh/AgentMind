"""
╔══════════════════════════════════════════════════════════════════╗
║              THE CORE: BUILDING THE LANGGRAPH AGENT              ║
║                                                                  ║
║  This file is where everything comes together.                   ║
║  We define the graph: nodes + edges = agent behavior.            ║
║                                                                  ║
║  LANGGRAPH GRAPH BUILDING STEPS:                                 ║
║  1. Define state (TypedDict)                                     ║
║  2. Create a StateGraph(StateClass)                              ║
║  3. Add nodes:  graph.add_node("name", function)                 ║
║  4. Add edges:  graph.add_edge("from", "to")                     ║
║  5. Add conditional edges for routing logic                      ║
║  6. Set entry point: graph.set_entry_point("first_node")         ║
║  7. Compile: app = graph.compile()                               ║
║  8. Run: app.invoke(initial_state) or app.stream(initial_state)  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
from dotenv import load_dotenv
from functools import partial

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START

from src.state import AgentState, create_initial_state
from src.nodes import agent_node, tools_node, route_after_agent
from tools.search import create_search_tool
from tools.calculator import create_calculator_tool
from tools.summarizer import create_summarizer_tool
from utils.display import console

load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# CONCEPT: TOOLS
# Tools are functions the agent can call to interact with the world.
# Each tool has:
#   - name: the string the LLM uses to call it
#   - description: tells the LLM WHEN and HOW to use it
#   - args_schema: Pydantic model defining the expected input
#
# The LLM reads the descriptions during inference and decides
# which tool (if any) to use based on the user's request.
# ─────────────────────────────────────────────────────────────────────

def create_tools():
    """
    Create all tools available to our agent.
    
    Tool selection strategy:
    - web_search: for finding current information
    - calculator: for math (LLMs are bad at arithmetic!)
    - summarize_text: for condensing long content
    
    Returning both a list (for LLM binding) and a dict (for lookup
    by name in tools_node).
    """
    search = create_search_tool()
    calculator = create_calculator_tool()
    summarizer = create_summarizer_tool()
    
    tools_list = [search, calculator, summarizer]
    tools_dict = {t.name: t for t in tools_list}
    
    return tools_list, tools_dict


# ─────────────────────────────────────────────────────────────────────
# CONCEPT: LLM SETUP
# We use Groq's free API to run Llama 3.1 (Meta's open-source model).
# 
# GROQ is a cloud service that runs open-source LLMs at high speed
# with a generous free tier — perfect for learning!
#
# bind_tools() is the key call:
# It tells the LLM about available tools by adding their schemas
# to the LLM's context/system prompt automatically.
# ─────────────────────────────────────────────────────────────────────

def create_llm(tools_list: list):
    """
    Initialize the LLM with tool-calling capability.
    
    We use llama-3.1-8b-instant — fast and free on Groq.
    For harder research tasks, llama-3.1-70b-versatile is better
    (slower but smarter).
    
    temperature=0: Makes the agent deterministic and consistent.
    For creative tasks you'd use 0.7-1.0.
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
    
    # bind_tools() is the magic:
    # It adds tool schemas to the LLM so it knows what tools exist
    # and how to call them. The LLM then outputs structured JSON
    # tool_calls when it wants to use a tool.
    llm_with_tools = llm.bind_tools(tools_list)
    
    console.print(f"[green]✓ LLM initialized:[/green] llama-3.1-8b-instant via Groq")
    return llm_with_tools


# ─────────────────────────────────────────────────────────────────────
# THE GRAPH — THE HEART OF THE AGENT
#
# CONCEPT: StateGraph
# StateGraph(AgentState) creates a graph where every node
# receives AgentState and returns a partial state update.
#
# Building blocks:
#   graph.add_node("name", fn)  — register a node
#   graph.add_edge("a", "b")    — always go from a to b
#   graph.add_conditional_edges("a", router_fn, {"key": "target"})
#                               — go from a to different nodes based
#                                 on what router_fn returns
# ─────────────────────────────────────────────────────────────────────

def build_agent_graph(llm_with_tools, tools_dict: dict):
    """
    Builds and compiles the LangGraph agent.
    
    The graph structure:
    
    START → agent_node → [conditional router]
                ↑              ↓ "use_tools"
           tools_node ──────────
                               ↓ "end"
                              END
    
    This creates the ReAct loop:
    Think → Act (tools) → Observe (tool results) → Think again → ...
    Until the agent decides it's done → END
    """
    
    # ── STEP 1: Create the graph with our state type ──────────────────
    # StateGraph takes the TypedDict class as the type parameter.
    # This tells LangGraph the shape of the state at compile time.
    graph = StateGraph(AgentState)
    
    # ── STEP 2: Create node functions with dependencies injected ──────
    # We use functools.partial to "bake in" the LLM and tools
    # because LangGraph node functions only receive (state) —
    # they can't have extra parameters.
    #
    # PATTERN: Dependency injection via partial
    # agent_fn = partial(agent_node, llm_with_tools=llm)
    # Now agent_fn(state) works even though agent_node needs llm too
    
    agent_fn = partial(agent_node, llm_with_tools=llm_with_tools)
    tools_fn = partial(tools_node, tools_by_name=tools_dict)
    
    # ── STEP 3: Register nodes ────────────────────────────────────────
    # "agent" and "tools" are just string names — they'll be referenced
    # in edge definitions below.
    graph.add_node("agent", agent_fn)
    graph.add_node("tools", tools_fn)
    
    # ── STEP 4: Set the entry point ───────────────────────────────────
    # This adds an edge from the special START node to "agent".
    # Equivalent to: graph.add_edge(START, "agent")
    graph.set_entry_point("agent")
    
    # ── STEP 5: Add conditional edges (the routing logic) ─────────────
    # After "agent" runs, call route_after_agent(state).
    # Its return value maps to the destination:
    #   "use_tools" → go to "tools" node
    #   "end"       → go to END (special termination node)
    graph.add_conditional_edges(
        "agent",                   # source node
        route_after_agent,         # routing function
        {
            "use_tools": "tools",  # route "use_tools" → tools node
            "end": END,            # route "end" → END (finish)
        }
    )
    
    # ── STEP 6: Add the loop-back edge ────────────────────────────────
    # After tools run, ALWAYS go back to agent.
    # This creates the ReAct loop: agent → tools → agent → tools → ...
    # The loop only breaks when route_after_agent returns "end"
    graph.add_edge("tools", "agent")
    
    # ── STEP 7: Compile ───────────────────────────────────────────────
    # compile() validates the graph and returns a runnable app.
    # At this point LangGraph checks:
    # - All edges reference valid nodes
    # - No dangling nodes
    # - Entry point is set
    #
    # OPTIONAL: Add a checkpointer for persistence
    # from langgraph.checkpoint.memory import MemorySaver
    # checkpointer = MemorySaver()  # in-memory persistence
    # app = graph.compile(checkpointer=checkpointer)
    # With a checkpointer, the agent can be paused and resumed!
    
    app = graph.compile()
    
    console.print("[green]✓ Agent graph compiled successfully[/green]")
    return app


# ─────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────

def create_agent():
    """Creates and returns a fully configured agent app."""
    tools_list, tools_dict = create_tools()
    llm_with_tools = create_llm(tools_list)
    app = build_agent_graph(llm_with_tools, tools_dict)
    return app


def run_agent(query: str, stream: bool = True) -> str:
    """
    Run the agent on a query and return the final answer.
    
    CONCEPT: invoke() vs stream()
    
    app.invoke(state) — runs the entire graph synchronously,
                        returns the final state
    
    app.stream(state) — runs the graph, yielding state snapshots
                        after each node. Perfect for showing
                        real-time progress to the user!
    
    For streaming, each yielded item is a dict:
    { "node_name": {partial_state_update} }
    """
    
    console.print(f"\n[bold blue]━━━ AgentMind Research Agent ━━━[/bold blue]")
    console.print(f"[bold]Query:[/bold] {query}\n")
    
    app = create_agent()
    initial_state = create_initial_state(query)
    
    final_state = None
    
    if stream:
        # ── STREAMING MODE ─────────────────────────────────────────
        # Each chunk tells us which node just ran and what it returned.
        # We can show live progress to the user.
        console.print("[dim]Streaming agent execution...[/dim]\n")
        
        for chunk in app.stream(initial_state):
            node_name = list(chunk.keys())[0]
            node_output = chunk[node_name]
            
            # Show which node just ran
            if node_name == "agent":
                pass  # Already shown inside agent_node
            elif node_name == "tools":
                pass  # Already shown inside tools_node
            
            # Keep track of latest state
            final_state = node_output
    else:
        # ── INVOKE MODE ─────────────────────────────────────────────
        # Runs everything and returns the final complete state.
        # Simpler but no live updates.
        final_state = app.invoke(initial_state)
    
    # Extract the final answer from the last AI message
    if final_state and "messages" in final_state:
        messages = final_state["messages"]
    else:
        # In streaming mode, rebuild from initial + stream
        final_full = app.invoke(initial_state)
        messages = final_full["messages"]
    
    # Get the last AIMessage (the agent's final response)
    final_answer = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                final_answer = msg.content
                break
    
    return final_answer
