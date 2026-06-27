"""
==================================================================
             UNIT TESTS: NODES & ROUTERS (pytest)                 
==================================================================

Run these tests in the terminal:
    pytest tests/test_agent_nodes.py
"""

import sys
import os
# Insert parent directory so we can run directly and import source code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from src.nodes import agent_node, tools_node, route_after_agent


# =====================================================================
# 1. TEST CONDITIONAL ROUTING EDGE
# =====================================================================

def test_route_after_agent():
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
    assert route_after_agent(state_tool) == "use_tools"
    
    # Case B: Last message is text only -> Route to 'end'
    state_end = {
        "messages": [
            AIMessage(content="The final answer is 4.")
        ]
    }
    assert route_after_agent(state_end) == "end"


# =====================================================================
# 2. TEST TOOLS NODE LOGIC
# =====================================================================

def test_tools_node():
    """
    Verifies that tools_node correctly invokes the calculator tool and 
    returns a list containing a ToolMessage with the correct tool_call_id.
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
    
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.content == "2 + 2 = 4"
    assert tool_msg.tool_call_id == "call-123"


# =====================================================================
# 3. TEST AGENT NODE WITH MOCKED LLM
# =====================================================================

def test_agent_node_with_mock_llm():
    """
    Verifies that agent_node invokes the bound LLM with the complete 
    message history, returns the response, and increments the count.
    """
    # Mock LLM to return a simple response
    mock_response = AIMessage(content="Mocked answer")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    
    # Setup initial mock state
    mock_state = {
        "messages": [HumanMessage(content="Hello")],
        "tool_call_count": 0
    }
    
    # Execute the node directly
    result = agent_node(mock_state, llm_with_tools=mock_llm)
    
    # Assert
    assert "messages" in result
    assert result["messages"] == [mock_response]
    assert result["tool_call_count"] == 1
    
    # Verify the LLM was invoked with the exact messages list
    mock_llm.invoke.assert_called_once_with(mock_state["messages"])
