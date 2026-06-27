from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage
from utils.display import console
from tools.search import create_search_tool
from functools import partial

# Custom list reducer for aggregating search findings
def append_findings(current: list[str] | None, new: list[str] | str | None) -> list[str]:
    current_list = list(current) if current is not None else []
    if new is None:
        return current_list
    if isinstance(new, list):
        return current_list + new
    return current_list + [new]


# State Definitions
class ResearcherState(TypedDict):
    """
    Isolated state for the child researcher subgraph.
    """
    topic: str
    queries: list[str]
    raw_results: Annotated[list[str], append_findings]
    final_summary: str


class SearchTask(TypedDict):
    """
    Input schema for the parallel search mapping tasks.
    """
    query: str


# Nodes
def generate_queries_node(state: ResearcherState, llm) -> dict:
    console.print("\n[bold cyan]🔎 [Subgraph] Planning search queries...[/bold cyan]")
    
    prompt = f"""You are a research planner. For the topic: '{state["topic"]}',
generate exactly 2 distinct, highly specific search query sentences.
Respond with only the queries, one query per line. Do not write numbers, bullet points, or introductory text."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    
    # Restrict to at most 2 queries for efficiency
    queries = queries[:2]
    
    console.print(f"  Generated Sub-queries: {queries}")
    return {"queries": queries}


def search_query_node(state: SearchTask, search_tool) -> dict:
    query = state["query"]
    console.print(f"[bold cyan]🔎 [Subgraph - Parallel Search] Searching for:[/bold cyan] [dim]{query}[/dim]")
    
    try:
        # Invoke search tool
        result = search_tool.invoke({"query": query, "num_results": 3})
        result_str = str(result)
    except Exception as e:
        result_str = f"Search failed: {e}"
        
    return {"raw_results": [f"Query: {query}\nResult:\n{result_str[:400]}..."]}


def summarize_findings_node(state: ResearcherState, llm) -> dict:
    console.print("\n[bold cyan]🔎 [Subgraph] Synthesizing parallel search findings...[/bold cyan]")
    
    joined_results = "\n\n=== RESULT ===\n".join(state["raw_results"])
    
    prompt = f"""Synthesize the following raw search results about the topic '{state["topic"]}'
into a concise summary report. Group the findings clearly:

{joined_results}

Write a summary of around 150 words."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_summary": response.content}


# Router for Send API
def route_to_searches(state: ResearcherState):
    """
    Map Phase: Map each query to a search_node Send task.
    """
    return [Send("search_node", {"query": q}) for q in state["queries"]]


# Subgraph Builder Function
def create_researcher_subgraph(llm):
    """
    Build and compile the researcher child subgraph.
    """
    search_tool = create_search_tool()
    
    builder = StateGraph(ResearcherState)
    
    builder.add_node("generate_queries", partial(generate_queries_node, llm=llm))
    builder.add_node("search_node", partial(search_query_node, search_tool=search_tool))
    builder.add_node("summarize_findings", partial(summarize_findings_node, llm=llm))
    
    builder.set_entry_point("generate_queries")
    
    # Conditional edge to trigger parallel searches
    builder.add_conditional_edges(
        "generate_queries",
        route_to_searches,
        ["search_node"]
    )
    
    # Join edge: go to summarize_findings once all parallel nodes complete
    builder.add_edge("search_node", "summarize_findings")
    builder.add_edge("summarize_findings", END)
    
    return builder.compile()
