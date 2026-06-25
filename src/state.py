"""
╔══════════════════════════════════════════════════════════════════╗
║                    CONCEPT: AGENT STATE                          ║
║                                                                  ║
║  In LangGraph, every node in your graph shares a single          ║
║  "state" object. Think of it as a whiteboard that every          ║
║  agent step can read from and write to.                          ║
║                                                                  ║
║  State is defined as a TypedDict — a Python dict with            ║
║  type hints for each field.                                      ║
║                                                                  ║
║  The key innovation: each field has an "annotation" that         ║
║  tells LangGraph HOW to update it when multiple nodes write.     ║
║  - add_messages: append new messages (don't overwrite)           ║
║  - default behavior: last-write wins (overwrite)                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The complete state of our research agent.
    
    This TypedDict is passed between EVERY node in the graph.
    Each node receives the full state and returns only the keys it updates.
    
    WHY TypedDict?
    - Type safety: Python can check you're not writing typos
    - Serializable: LangGraph can save/restore this to a database
    - Reducers: The Annotated[..., add_messages] syntax tells LangGraph
      how to MERGE updates from parallel nodes or sequential writes
    """
    
    # ── CONVERSATION HISTORY ─────────────────────────────────────────
    # Annotated[list, add_messages] is CRUCIAL:
    #   - Without it: each node would OVERWRITE the message list
    #   - With add_messages: messages are APPENDED (accumulated)
    # This is the standard pattern for all LangGraph chat agents.
    messages: Annotated[list, add_messages]
    
    # ── AGENT METADATA ────────────────────────────────────────────────
    # The original user query — stored separately so we never lose it
    # even as the messages list grows with tool calls and observations
    original_query: str
    
    # How many tool calls have been made? Used to prevent infinite loops.
    # If the agent calls tools 10+ times, we force it to conclude.
    tool_call_count: int
    
    # ── RESEARCH ACCUMULATION ─────────────────────────────────────────
    # As the agent searches the web, findings are collected here.
    # The final node reads these to write the summary report.
    research_findings: list[str]
    
    # Has the agent decided it has enough info to answer?
    # The conditional edge checks this to decide: loop again or END
    is_complete: bool
    
    # ── FINAL OUTPUT ──────────────────────────────────────────────────
    # The polished final answer written by the agent
    final_report: str


# ── DEFAULT STATE FACTORY ─────────────────────────────────────────────
def create_initial_state(query: str) -> AgentState:
    """
    Creates the initial state for a new research session.
    
    We import SystemMessage and HumanMessage from langchain_core.
    These are the building blocks of the messages list:
    
    - SystemMessage: Instructions to the LLM (its "personality" and rules)
    - HumanMessage: What the user said
    - AIMessage: What the AI responded
    - ToolMessage: The result of a tool call
    
    LangGraph's add_messages reducer handles all of this automatically.
    """
    from langchain_core.messages import HumanMessage
    from utils.prompts import SYSTEM_PROMPT
    from langchain_core.messages import SystemMessage
    
    return AgentState(
        messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ],
        original_query=query,
        tool_call_count=0,
        research_findings=[],
        is_complete=False,
        final_report="",
    )
