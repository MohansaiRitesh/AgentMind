"""
==================================================================
             LEARNING LAB: NESTED SUBGRAPHS & MAP-REDUCE          
==================================================================

In this lab, you will learn how to:
1. Isolate agent variables using nested child subgraphs.
2. Dynamically execute tasks in parallel using the Send API (Map phase).
3. Join and reduce parallel outcomes back into a single state (Reduce phase).
4. Run a parent graph coordinating child execution flows.

Run this script:
    python examples/hierarchical_subgraphs.py
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
from langgraph.types import Send

# Import search tool from workspace
from tools.search import create_search_tool
from utils.display import console

# Load environment variables
load_dotenv()


# =====================================================================
# CUSTOM LIST REDUCER
# =====================================================================

def append_findings(current: list[str] | None, new: list[str] | str | None) -> list[str]:
    """ Custom reducer to accumulate raw search results from parallel nodes. """
    current_list = list(current) if current is not None else []
    if new is None:
        return current_list
    if isinstance(new, list):
        return current_list + new
    return current_list + [new]


# =====================================================================
# 1. STATE SCHEMAS (ISOLATION)
# =====================================================================

class ResearcherState(TypedDict):
    """ Isolated child state. Parent cannot see these search steps. """
    topic: str
    queries: list[str]
    raw_results: Annotated[list[str], append_findings]  # Reducer gathers parallel inputs
    final_summary: str


class ParentState(TypedDict):
    """ Parent global state. Contains final report and intermediate summaries. """
    messages: Annotated[list, add_messages]
    research_findings: list[str]
    final_report: str


# =====================================================================
# 2. CHILD SUBGRAPH NODES
# =====================================================================

def generate_queries_node(state: ResearcherState, llm) -> dict:
    """ Generates 2 sub-queries to map-reduce. """
    console.print("\n[Node: Subgraph - generate_queries] Planning search queries...")
    
    prompt = f"""You are a research planner. For the topic: '{state["topic"]}',
generate exactly 2 distinct, highly specific search query sentences.
Respond with only the queries, one query per line. Do not write numbers, bullet points, or introductory text."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    
    # Slice to ensure we only have 2 queries max for demonstration efficiency
    queries = queries[:2]
    
    console.print(f"  Generated Sub-queries: {queries}")
    return {"queries": queries}


# The Map Task input state schema for Send API
class SearchTask(TypedDict):
    query: str


def search_query_node(state: SearchTask, search_tool) -> dict:
    """
    Executes a single search task in parallel.
    Notice that 'state' is the local SearchTask dictionary sent by the Send API,
    NOT the global ResearcherState.
    """
    query = state["query"]
    console.print(f"[Node: Subgraph - search_node (Parallel Map)] Searching for: [cyan]{query}[/cyan]")
    
    try:
        result = search_tool.invoke({"query": query, "num_results": 2})
        result_str = str(result)
    except Exception as e:
        result_str = f"Search failed: {e}"
        
    # Return delta update for ResearcherState. raw_results has append_findings reducer,
    # so updates from parallel search nodes will be merged into a single list.
    return {"raw_results": [f"Query: {query}\nResult:\n{result_str[:400]}..."]}


def summarize_findings_node(state: ResearcherState, llm) -> dict:
    """
    Reduce node: Joins all parallel search outputs and compiles a summary.
    This executes only after all search_query_nodes complete.
    """
    console.print("\n[Node: Subgraph - summarize_findings (Reduce)] Synthesizing findings...")
    
    joined_results = "\n\n=== RESULT ===\n".join(state["raw_results"])
    
    prompt = f"""Synthesize the following raw search results about the topic '{state["topic"]}'
into a concise summary report. Group the findings clearly:

{joined_results}

Write a summary of around 150 words."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_summary": response.content}


# =====================================================================
# 3. DYNAMIC SEND ROUTER (MAP-REDUCE)
# =====================================================================

def route_to_searches(state: ResearcherState):
    """
    Conditional edge: Map phase.
    For each query in state['queries'], returns a Send object targeting 'search_node'.
    """
    # Send(target_node_name, local_node_input)
    return [Send("search_node", {"query": q}) for q in state["queries"]]


# =====================================================================
# 4. PARENT GRAPH NODES
# =====================================================================

def researcher_node(state: ParentState, compiled_subgraph) -> dict:
    """
    Parent Node calling the nested subgraph.
    """
    console.print("\n[Node: Parent - researcher_node] Starting Child Subgraph...")
    
    # Extract latest human message content
    topic = state["messages"][-1].content
    
    # Invoke child subgraph as a function, supplying its initial state input
    subgraph_output = compiled_subgraph.invoke({"topic": topic})
    
    console.print("[Node: Parent - researcher_node] Child Subgraph completed successfully.")
    
    # Map child final summary back to parent state
    return {"research_findings": [subgraph_output["final_summary"]]}


def writer_node(state: ParentState, llm) -> dict:
    """ Parent Node compiling the final markdown report. """
    console.print("\n[Node: Parent - writer_node] Generating final comprehensive report...")
    
    findings = "\n\n".join(state["research_findings"])
    
    prompt = f"""Write a polished, professional research report in markdown format.
Use the following findings:

{findings}

Structure the report with headers, key findings, and a final conclusion."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": response.content}


# =====================================================================
# 5. BUILD AND COMPILE BOTH GRAPHS
# =====================================================================

def build_hierarchical_agent():
    # Setup LLM & Tool
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment.")
        
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=api_key)
    search_tool = create_search_tool()
    
    # ── A. COMPILE CHILD SUBGRAPH ────────────────────────────────────
    sub_builder = StateGraph(ResearcherState)
    
    # Bind nodes
    from functools import partial
    sub_builder.add_node("generate_queries", partial(generate_queries_node, llm=llm))
    sub_builder.add_node("search_node", partial(search_query_node, search_tool=search_tool))
    sub_builder.add_node("summarize_findings", partial(summarize_findings_node, llm=llm))
    
    # Setup routing connectivity
    sub_builder.set_entry_point("generate_queries")
    
    # Send queries to search_node in parallel
    sub_builder.add_conditional_edges(
        "generate_queries",
        route_to_searches,
        ["search_node"]  # Tells compiler it will map to search_node
    )
    
    # Map-Reduce join: go from search_node to summarize_findings once all finish
    sub_builder.add_edge("search_node", "summarize_findings")
    sub_builder.add_edge("summarize_findings", END)
    
    compiled_subgraph = sub_builder.compile()
    
    # ── B. COMPILE PARENT GRAPH ──────────────────────────────────────
    parent_builder = StateGraph(ParentState)
    
    # Bind nodes (injecting child subgraph as a dependency)
    parent_builder.add_node("researcher", partial(researcher_node, compiled_subgraph=compiled_subgraph))
    parent_builder.add_node("writer", partial(writer_node, llm=llm))
    
    # Parent connectivity
    parent_builder.set_entry_point("researcher")
    parent_builder.add_edge("researcher", "writer")
    parent_builder.add_edge("writer", END)
    
    return parent_builder.compile()


# =====================================================================
# 6. RUNNER DEMO
# =====================================================================

def main():
    console.print("[bold blue]=== LangGraph Learning Lab: Hierarchical Subgraphs & Map-Reduce ===[/bold blue]")
    
    app = build_hierarchical_agent()
    
    query = "Compare Tesla and BYD EV sales trends in 2024."
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    console.print(f"[bold]Query:[/bold] {query}")
    console.print("[dim]Invoking Parent Graph...[/dim]")
    
    final_state = app.invoke(initial_state)
    
    console.print("\n[bold green]=== Parent Graph Execution Output ===[/bold green]")
    console.print("\n[bold]Isolated findings compiled by nested child graph:[/bold]")
    console.print(final_state["research_findings"][0])
    
    console.print("\n[bold]Final Markdown Report by Parent Graph:[/bold]")
    console.print(final_state["final_report"])


if __name__ == "__main__":
    main()
