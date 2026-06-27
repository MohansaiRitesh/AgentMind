from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

# Custom reducers
def sum_tokens(current: int | None, new: int | None) -> int:
    """A custom reducer function that adds two integers."""
    current_val = current if current is not None else 0
    new_val = new if new is not None else 0
    return current_val + new_val


def append_logs(current: list[str] | None, new: list[str] | str | None) -> list[str]:
    """A custom reducer function to compile a running list of strings."""
    current_logs = list(current) if current is not None else []
    if new is None:
        return current_logs
    if isinstance(new, list):
        return current_logs + new
    return current_logs + [new]


class AgentState(TypedDict):
    """
    The complete state of our research agent.
    """
    
    # ── CONVERSATION HISTORY ─────────────────────────────────────────
    messages: Annotated[list, add_messages]
    
    # ── AGENT METADATA ────────────────────────────────────────────────
    original_query: str
    tool_call_count: int
    
    # ── RESEARCH ACCUMULATION ─────────────────────────────────────────
    research_findings: Annotated[list[str], append_logs]
    is_complete: bool
    
    # ── FINAL OUTPUT ──────────────────────────────────────────────────
    final_report: str
    
    # ── ADVANCED LOGGING & METRICS & SAFETY ───────────────────────────
    prompt_tokens: Annotated[int, sum_tokens]
    completion_tokens: Annotated[int, sum_tokens]
    total_tokens: Annotated[int, sum_tokens]
    execution_logs: Annotated[list[str], append_logs]
    is_approved: bool


# ── DEFAULT STATE FACTORY ─────────────────────────────────────────────
def create_initial_state(query: str) -> AgentState:
    """
    Creates the initial state for a new research session.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.prompts import SYSTEM_PROMPT
    
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
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        execution_logs=["Session initialized."],
        is_approved=False,
    )
