# 🤖 AgentMind — Advanced Agentic AI Research Agent

A production-ready, highly advanced LangGraph-powered research agent featuring nested child subgraphs, parallel Map-Reduce execution, state-level custom reducers, human-in-the-loop safety gates, and SQLite-backed persistence.

---

## 🚀 Advanced Architecture & Core Concepts

AgentMind implements a modular, persistent architecture integrating modern agentic patterns:

```
                       User Query
                           │
                           ▼
                 [AgentMind Parent Graph]
                           │
                           ▼
                    researcher_node
               (Invokes Child Subgraph)
              /                        \
             ▼                          ▼
       search_query_node          search_query_node
       (Parallel Search)          (Parallel Search)
             \                          /
              ▼                        ▼
                   summarize_findings
                   (Reduce Synthesis)
                           │
                           ▼
                      agent_node  ◄───┐
                    (LLM reasoning)   │
                           │          │
                           ▼          │
                    validator_node    │
                (Large values gate)   │
                           │          │
                    [NodeInterrupt]   │
               (Interrupt menu pauses)│
                           │          │
                    [Conditional]     │
                     /          \     │
        (use_tools) ▼            ▼    │
               tools_node ───────┘    │
                                      │
                                 (end)▼
                                     END
```

1. **State Reducers & Token Tracking**: Custom channel summation reducers (`sum_tokens`) track input, output, and total tokens across all LLM steps. The list accumulation reducer (`append_logs`) aggregates sequential execution logs.
2. **Hierarchical Subgraphs**: Extract search planning and gathering tasks into an isolated child subgraph (`src/researcher_graph.py`), shielding parent state from raw search results.
3. **Map-Reduce Parallelism**: Uses LangGraph's `Send` API to map generated sub-queries to parallel execution nodes (`search_query_node`), joining findings on a downstream reduction node.
4. **Safety Gating & Interrupts**: A validator node intercepts tool calls and raises `NodeInterrupt` if calculator values exceed `1000`, pausing execution and saving the current checkpoint.
5. **Interactive CLI Dashboard**: The main execution loop catches interrupts and displays a menu asking the user to:
   - **Approve**: authorize the pending action as-is.
   - **Edit arguments**: modify tool arguments manually before continuing.
   - **Skip & Mock**: bypass tool execution entirely and mock tool results.
   - **Terminate**: abort execution and leave checkpoint saved.
6. **SQLite Persistent Saving**: Compiles the graph with `SqliteSaver` to persist thread session checkpoints durably in `agent_memory.db`.

---

## 🛠️ Free Stack Requirements

| Dependency | Purpose | Cost |
|------------|---------|------|
| LangGraph | Graph state orchestration | Free |
| langgraph-checkpoint-sqlite | SQLite checkpoint storage | Free |
| Groq + Llama 3.1 | Reasoning LLM brain | Free |
| DuckDuckGo Search | Web search tool | Free |

---

## 📦 Installation & Setup

1. Clone the repository and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Get your free Groq API key at https://console.groq.com.
3. Create a `.env` file at the root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## 🚀 Execution & Run Modes

AgentMind is run from `main.py` using PowerShell or standard command line:

```bash
# Set console encoding to UTF-8 on Windows for clean box border styling
$env:PYTHONIOENCODING="utf-8"

# 1. Run safety interrupt demo query (Calculate 5000 * 2)
python main.py

# 2. Start the interactive console research loops
python main.py --interactive

# 3. Query the agent directly
python main.py --query "Compare EV sales trends in 2024."
```

*Note: In interactive mode, switch active SQLite database thread sessions by typing `thread <id>`.*

---

## 🧪 Testing

We verify node routing updates, token parsing, and validator gate interrupts using `pytest`:
```bash
python -m pytest tests/test_agent_nodes.py
```

---

## 📁 Repository Structure

```
agentmind/
├── main.py                   # CLI loop, SQLite checkpointer setup & interrupt handling
├── langgraph.json            # Studio configuration
├── src/
│   ├── agent.py              # Parent graph creation and configuration builder
│   ├── state.py              # Custom reducers and TypedDict schema channels
│   ├── nodes.py              # Parent node implementations (agent, validator, tools)
│   └── researcher_graph.py   # Isolated child researcher subgraph using Send API
├── tools/
│   ├── search.py             # DuckDuckGo search tool
│   ├── calculator.py         # Computational tool
│   └── summarizer.py         # Condensing tool
├── utils/
│   ├── display.py            # Pretty rich-text display console styling
│   └── prompts.py            # LLM system configurations
├── tests/
│   └── test_agent_nodes.py   # Unit test suite verifying router, validator & node inputs
└── requirements.txt          # Python dependencies
```
