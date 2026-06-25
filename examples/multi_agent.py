"""
╔══════════════════════════════════════════════════════════════════╗
║              ADVANCED CONCEPT: MULTI-AGENT SYSTEMS               ║
║                                                                  ║
║  Real-world tasks often need specialized agents working          ║
║  together. LangGraph supports this via subgraphs and routing.    ║
║                                                                  ║
║  This example implements the SUPERVISOR pattern:                 ║
║                                                                  ║
║       User Query                                                 ║
║           ↓                                                      ║
║      [Supervisor LLM]  ← decides which specialist to use         ║
║       /    |    \                                                ║
║   [Researcher] [Writer] [Analyst]  ← specialist agents           ║
║       \    |    /                                                ║
║      [Supervisor LLM]  ← reviews output, routes again or FINISH  ║
║           ↓                                                      ║
║        [END]                                                     ║
╚══════════════════════════════════════════════════════════════════╝

Run: python examples/multi_agent.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.display import console


# ─────────────────────────────────────────────────────────────────────
# MULTI-AGENT STATE
# Extended state tracking which agent did what
# ─────────────────────────────────────────────────────────────────────

class MultiAgentState(dict):
    messages: Annotated[list, add_messages]
    next_agent: str          # Which specialist to call next
    research_done: bool      # Has research been completed?
    analysis_done: bool      # Has analysis been completed?
    final_answer: str        # The assembled final response


# ─────────────────────────────────────────────────────────────────────
# SPECIALIST AGENTS
# Each is a simple node with a focused system prompt
# ─────────────────────────────────────────────────────────────────────

def make_specialist(role: str, instructions: str, llm):
    """Factory function that creates specialist agent nodes."""
    system = SystemMessage(content=f"You are a specialist {role}. {instructions}")
    
    def specialist_node(state):
        console.print(f"\n[bold cyan]🤖 {role} agent working...[/bold cyan]")
        messages = [system] + state["messages"]
        response = llm.invoke(messages)
        console.print(f"  [green]✓ {role} done[/green]")
        return {"messages": [response]}
    
    specialist_node.__name__ = role.lower().replace(" ", "_") + "_node"
    return specialist_node


def make_supervisor(llm, agents: list[str]):
    """
    Creates the supervisor node.
    
    The supervisor's job:
    1. Read all messages so far (including specialist outputs)
    2. Decide: call another specialist, or we're done (FINISH)?
    3. Return the routing decision
    
    We use structured output (response_format or tool calling) to get
    a clean routing decision rather than parsing free text.
    """
    options = agents + ["FINISH"]
    
    supervisor_prompt = f"""You are a research supervisor coordinating specialist agents.
Available agents: {', '.join(agents)}

Based on the conversation so far, decide what to do next:
- If more research is needed → respond with exactly: RESEARCHER
- If data analysis is needed → respond with exactly: ANALYST  
- If a final written report is needed → respond with exactly: WRITER
- If the task is complete → respond with exactly: FINISH

Respond with ONLY the agent name or FINISH, nothing else."""
    
    system = SystemMessage(content=supervisor_prompt)
    
    def supervisor_node(state):
        console.print(f"\n[bold purple]👔 Supervisor deciding...[/bold purple]")
        messages = [system] + state["messages"]
        response = llm.invoke(messages)
        
        decision = response.content.strip().upper()
        # Clean up in case model adds extra text
        for option in options:
            if option.upper() in decision:
                decision = option.upper()
                break
        else:
            decision = "FINISH"  # Default to finish if unclear
        
        console.print(f"  [yellow]→ Routing to: {decision}[/yellow]")
        return {"next_agent": decision, "messages": [response]}
    
    return supervisor_node


def route_from_supervisor(state) -> str:
    """Conditional edge: reads next_agent from state."""
    return state.get("next_agent", "FINISH")


# ─────────────────────────────────────────────────────────────────────
# BUILD THE MULTI-AGENT GRAPH
# ─────────────────────────────────────────────────────────────────────

def build_multi_agent_system():
    api_key = os.getenv("GROQ_API_KEY")
    # Use a smarter model for the supervisor
    supervisor_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    specialist_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=api_key)
    
    # Create specialist agents
    researcher = make_specialist(
        "RESEARCHER",
        "Find and present factual information on the topic. Be thorough and cite what you know.",
        specialist_llm
    )
    analyst = make_specialist(
        "ANALYST",
        "Analyze the research provided. Find patterns, implications, and key insights.",
        specialist_llm
    )
    writer = make_specialist(
        "WRITER",
        "Write a clear, well-structured final report based on the research and analysis. Use markdown headers.",
        specialist_llm
    )
    
    supervisor = make_supervisor(supervisor_llm, ["RESEARCHER", "ANALYST", "WRITER"])
    
    # Build the graph
    from typing import TypedDict
    
    # Use a simpler TypedDict for multi-agent state
    from typing import Annotated
    from langgraph.graph.message import add_messages
    
    class State(dict):
        pass
    
    graph = StateGraph(dict)
    
    # Add all nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("RESEARCHER", researcher)
    graph.add_node("ANALYST", analyst)
    graph.add_node("WRITER", writer)
    
    # Supervisor is the entry point
    graph.set_entry_point("supervisor")
    
    # After supervisor decides, route to the chosen specialist or END
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "RESEARCHER": "RESEARCHER",
            "ANALYST": "ANALYST",
            "WRITER": "WRITER",
            "FINISH": END,
        }
    )
    
    # After any specialist, ALWAYS return to supervisor
    # Supervisor reviews the output and decides what's next
    graph.add_edge("RESEARCHER", "supervisor")
    graph.add_edge("ANALYST", "supervisor")
    graph.add_edge("WRITER", "supervisor")
    
    return graph.compile()


def run_multi_agent(query: str):
    """Run the multi-agent system on a query."""
    console.print(f"\n[bold blue]━━━ Multi-Agent System ━━━[/bold blue]")
    console.print(f"[bold]Query:[/bold] {query}\n")
    
    app = build_multi_agent_system()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "",
        "final_answer": "",
    }
    
    final_state = app.invoke(initial_state)
    
    # Get the last substantive message (from WRITER)
    messages = final_state["messages"]
    for msg in reversed(messages):
        if hasattr(msg, "content") and len(msg.content) > 100:
            console.print(f"\n[bold green]Final Report:[/bold green]")
            console.print(msg.content)
            break


if __name__ == "__main__":
    run_multi_agent("Explain the key differences between supervised and unsupervised machine learning, with examples.")
