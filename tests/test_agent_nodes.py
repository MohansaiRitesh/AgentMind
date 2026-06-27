"""
==================================================================
             UNIT TESTS: NODES & ROUTERS (pytest)                 
==================================================================

Run these tests in the terminal:
    python -m pytest tests/test_agent_nodes.py
"""

import sys
import os
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.errors import NodeInterrupt

# Insert parent directory so we can run directly and import source code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nodes import agent_node, tools_node, validator_node, route_after_validator


# =====================================================================
# 1. TEST CONDITIONAL ROUTING EDGE
# =====================================================================

def test_route_after_validator():
    """
    Verifies that the routing edge checks if the last AIMessage 
    contains tool calls and correctly routes to 'use_tools' or 'end'.
    """
    # Case A: Last message contains tool calls -> Route to 'use_tools'
    state_tool = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "tc1"}]
            )
        ]
    }
    assert route_after_validator(state_tool) == "use_tools"
    
    # Case B: Last message is text only -> Route to 'end'
    state_end = {
        "messages": [
            AIMessage(content="The final answer is 4.")
        ]
    }
    assert route_after_validator(state_end) == "end"


# =====================================================================
# 2. TEST TOOLS NODE LOGIC
# =====================================================================

def test_tools_node():
    """
    Verifies that tools_node executes tool and accumulates logs.
    """
    # Create a mock calculator tool
    mock_tool = MagicMock()
    mock_tool.name = "calculator"
    mock_tool.invoke.return_value = "2 + 2 = 4"
    tools_by_name = {"calculator": mock_tool}
    
    # Set the state representing the requested tool call
    mock_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "calculator", "args": {"expression": "2 + 2"}, "id": "call-123"}]
            )
        ],
        "research_findings": []
    }
    
    # Execute the node directly
    result = tools_node(mock_state, tools_by_name=tools_by_name)
    
    # Assert
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "execution_logs" in result
    
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.content == "2 + 2 = 4"
    assert tool_msg.tool_call_id == "call-123"
    assert any("executed" in log for log in result["execution_logs"])


# =====================================================================
# 3. TEST AGENT NODE WITH MOCKED LLM
# =====================================================================

def test_agent_node_with_mock_llm():
    """
    Verifies that agent_node invokes LLM, increments count, and extracts tokens.
    """
    # Mock LLM to return a response with metadata
    mock_response = AIMessage(content="Mocked answer")
    mock_response.response_metadata = {
        "token_usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    
    # Setup initial mock state
    mock_state = {
        "messages": [HumanMessage(content="Hello")],
        "tool_call_count": 0,
        "research_findings": []
    }
    
    # Execute the node directly
    result = agent_node(mock_state, llm_with_tools=mock_llm)
    
    # Assert
    assert "messages" in result
    assert result["messages"] == [mock_response]
    assert result["tool_call_count"] == 1
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 5
    assert result["total_tokens"] == 15
    assert len(result["execution_logs"]) == 1


# =====================================================================
# 4. TEST VALIDATOR NODE SAFETY GATES
# =====================================================================

def test_validator_node_safety_gates():
    """
    Verifies that validator_node intercept rules work:
    - Safe calculation passes.
    - Large calculation raises NodeInterrupt if unapproved.
    - Large calculation passes if approved.
    """
    # Safe calculator expression (<= 1000)
    state_safe = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "calculator", "args": {"expression": "100 * 5"}, "id": "t-1"}]
            )
        ],
        "is_approved": False
    }
    # Should not raise any exception
    res_safe = validator_node(state_safe)
    assert "execution_logs" in res_safe
    
    # Large calculator expression (> 1000) - Unapproved
    state_unsafe_unapproved = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "calculator", "args": {"expression": "2000 * 3"}, "id": "t-2"}]
            )
        ],
        "is_approved": False
    }
    with pytest.raises(BaseException) as exc_info:
        validator_node(state_unsafe_unapproved)
    assert "Safety Check Required" in str(exc_info.value)
    
    # Large calculator expression (> 1000) - Approved
    state_unsafe_approved = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "calculator", "args": {"expression": "2000 * 3"}, "id": "t-3"}]
            )
        ],
        "is_approved": True
    }
    # Should bypass safety gate and complete
    res_approved = validator_node(state_unsafe_approved)
    assert "execution_logs" in res_approved
