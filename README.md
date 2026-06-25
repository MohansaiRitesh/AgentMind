# 🤖 AgentMind — Agentic AI Research Agent

A complete LangGraph-powered research agent that plans, searches, reasons, and reports.

## 🧠 What You'll Learn

1. **Agentic AI fundamentals** — agents, tools, loops, state
2. **LangGraph core concepts** — nodes, edges, state, conditional routing
3. **ReAct pattern** — Reason → Act → Observe loop
4. **Tool use** — how agents call external tools
5. **Memory & State** — persisting context across steps
6. **Human-in-the-loop** — interrupting execution for approval

## 🏗️ Architecture

```
               User Query
                  ↓
          [AgentMind Graph]
                  ↓
 ┌──────────────────────────────────────┐
 │  START → agent_node → [conditional]  │
 │              ↑              ↓        │
 │         tools_node ←── use_tools?    │
 │                           ↓ no       │
 │                          END         │
 └──────────────────────────────────────┘
```

## 🛠️ Free Tools Used

| Tool | Purpose | Cost |
|------|---------|------|
| LangGraph | Agent orchestration | Free |
| Groq + Llama 3.1 | LLM brain | Free tier |
| DuckDuckGo Search | Web search | Free |
| LangChain Community | Tool integrations | Free |

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Setup

1. Get a **free** Groq API key at https://console.groq.com (no credit card needed)
2. Copy `.env.example` to `.env`
3. Add your `GROQ_API_KEY`

```bash
cp .env.example .env
# Edit .env and add your key
```

## 🚀 Run

```bash
# Basic run
python main.py

# Interactive mode
python main.py --interactive

# Run specific query
python main.py --query "What are the latest breakthroughs in quantum computing?"
```

## 📁 Project Structure

```
agentmind/
├── main.py              # Entry point
├── src/
│   ├── agent.py         # Core LangGraph agent (THE MAIN FILE)
│   ├── state.py         # AgentState definition
│   └── nodes.py         # All graph nodes (agent, tools)
├── tools/
│   ├── search.py        # DuckDuckGo web search tool
│   ├── calculator.py    # Math computation tool
│   └── summarizer.py    # Text summarization tool
├── utils/
│   ├── display.py       # Pretty terminal output
│   └── prompts.py       # System prompts for the agent
├── requirements.txt
├── .env.example
└── README.md
```

## 🎓 Key Concepts Explained

### 1. AgentState
The shared "memory" passed between all nodes in the graph.
Every node reads from it and writes back to it.

### 2. Nodes
Python functions that take state → return updated state.
- `agent_node`: Calls the LLM to decide what to do next
- `tools_node`: Executes whatever tool the LLM chose

### 3. Edges
Connections between nodes.
- Normal edges: always go A → B
- Conditional edges: route based on logic ("if LLM called a tool → go to tools_node, else → END")

### 4. ReAct Pattern
**Reason** + **Act** — the agent thinks step by step:
1. **Thought**: "I need to find recent news about X"
2. **Action**: Call search tool with query "X news 2024"
3. **Observation**: Gets search results back
4. **Thought**: "Now I have enough info to answer"
5. **Answer**: Produces final response
