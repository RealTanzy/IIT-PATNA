# Chapter 7: AI Agents — Autonomous Systems

> "The best tool is the one that knows when to use another tool."

---

## 7.1 What Is an AI Agent?

An AI agent is a software system that perceives its environment, reasons about what to do, and takes actions to achieve a goal — autonomously, without step-by-step human direction.

### The Core Loop: Perception → Reasoning → Action

Every agent, regardless of architecture, follows a fundamental cycle:

```
while goal_not_achieved:
    observation = perceive(environment)
    thought = reason(observation, memory, goal)
    action = decide(thought)
    result = execute(action)
    memory.update(result)
```

**Perception** is how the agent receives information — user messages, tool outputs, sensor data, API responses. **Reasoning** is the cognitive step where the LLM processes context and decides what to do next. **Action** is the execution of a tool call, an API request, or a response to the user.

### Chatbots vs. Agents

| Dimension | Chatbot | Agent |
|-----------|---------|-------|
| State | Stateless (each turn independent) | Stateful (remembers across turns) |
| Tools | None | Uses external tools |
| Goal | Answer the current question | Achieve a multi-step objective |
| Autonomy | Reactive only | Proactive, can plan ahead |
| Control flow | Linear (input → output) | Looped (perceive → reason → act → observe) |

A chatbot answers "What is the weather?" An agent books your flight based on your calendar, the weather forecast, and your airline preferences — calling multiple APIs along the way.

### Types of Agent Architectures

**ReAct (Reason + Act):** The agent alternates between thinking and acting. Each thought explicitly reasons about what to do next, then a tool is called, and the observation feeds the next thought. Simple, effective, widely used.

**Plan-and-Execute:** The agent first creates a complete plan (a list of steps), then executes them sequentially. Good for well-defined tasks where the full path is knowable upfront. Less adaptive to surprises.

**Reflexion:** The agent attempts a task, evaluates its own performance, generates self-critique, and retries with improved strategy. Adds a meta-cognitive layer. Useful when the first attempt often fails (e.g., code generation with unit tests).

---

## 7.2 The ReAct Pattern

ReAct (Yao et al., 2022) interleaves reasoning traces with action execution. The key insight: letting the model "think out loud" before acting dramatically improves tool selection and reduces hallucination.

### The Pattern

```
Thought: I need to find the root cause of the signal failure.
         The signal SIG_BRK_PRESSURE disappeared after commit abc123.
         Let me check what changed in that commit.

Action: git_diff(commit="abc123", file="brake_module.c")

Observation: Lines 45-52 were deleted. These lines contained the
             signal publication for SIG_BRK_PRESSURE.

Thought: The signal was deleted in commit abc123. But I should verify
         this wasn't moved to another file. Let me search the codebase.

Action: search_codebase(query="SIG_BRK_PRESSURE", scope="current")

Observation: No results found in current codebase.

Thought: The signal definition was deleted and not relocated.
         Root cause identified: code deletion in commit abc123.

Action: return_result(cause="CODE_DELETED", confidence=0.95)
```

### Pseudocode Implementation

```python
def react_agent(goal: str, tools: list[Tool], max_steps: int = 10):
    messages = [SystemMessage(REACT_PROMPT), HumanMessage(goal)]
    
    for step in range(max_steps):
        # Reasoning step: LLM generates thought + action
        response = llm.invoke(messages, tools=tools)
        
        if response.is_final_answer:
            return response.content
        
        # Action step: execute the selected tool
        tool_call = response.tool_calls[0]
        tool = find_tool(tool_call.name, tools)
        observation = tool.execute(**tool_call.arguments)
        
        # Feed observation back
        messages.append(AIMessage(response))
        messages.append(ToolMessage(observation, tool_call_id=tool_call.id))
    
    return "Max steps reached without conclusion"
```

### When to Use ReAct vs. Simpler Patterns

| Scenario | Pattern |
|----------|---------|
| Single API call needed | Direct function call (no agent needed) |
| Linear multi-step task | Plan-and-Execute |
| Exploratory investigation | ReAct |
| Task with uncertain path | ReAct |
| Self-improving code gen | Reflexion |
| High-stakes with review | ReAct + human-in-the-loop |

---

## 7.3 Tool-Use & Function Calling

Tools are how agents interact with the world beyond text generation. The LLM doesn't execute tools itself — it generates a structured request, and the runtime executes it.

### Tool Schema Definition

Every tool is defined by a JSON Schema:

```json
{
  "name": "get_signal_dependencies",
  "description": "Returns all upstream and downstream dependencies for a given signal in the dependency graph. Use this when you need to trace signal flow.",
  "parameters": {
    "type": "object",
    "properties": {
      "signal_name": {
        "type": "string",
        "description": "The full signal name (e.g., 'SIG_BRK_PRESSURE_FL')"
      },
      "direction": {
        "type": "string",
        "enum": ["upstream", "downstream", "both"],
        "description": "Which direction to traverse the dependency graph"
      },
      "max_depth": {
        "type": "integer",
        "default": 3,
        "description": "Maximum traversal depth"
      }
    },
    "required": ["signal_name"]
  }
}
```

The **description** is critical — it tells the LLM *when* to use the tool. Poor descriptions lead to incorrect tool selection.

### How LLMs Select Tools

The model sees all available tool schemas in its context and selects based on:
1. **Description matching** — does the tool's purpose match the current need?
2. **Parameter fit** — do available data points match required parameters?
3. **Context** — what did previous tool calls return?

**Parallel tool calls:** Modern LLMs can request multiple tools in a single turn when the calls are independent:

```python
# The model might generate two parallel calls:
tool_calls = [
    {"name": "get_git_diff", "args": {"commit": "abc123"}},
    {"name": "get_signal_config", "args": {"signal": "SIG_BRK"}}
]
```

**Forced tool use:** You can constrain the model to always use a specific tool:

```python
response = llm.invoke(
    messages,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_signal_dependencies"}
)
```

### Error Handling

Tools fail. Networks time out. APIs return errors. A robust agent handles this gracefully:

```python
def execute_tool_safely(tool, arguments, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = tool.execute(**arguments)
            return ToolResult(success=True, data=result)
        except TimeoutError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            return ToolResult(success=False, error="Tool timed out after retries")
        except ValidationError as e:
            # Don't retry validation errors — fix the input
            return ToolResult(success=False, error=f"Invalid params: {e}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

The agent sees the error in the observation and can reason about what to try differently.

### Safety: The Gatekeeper Pattern

For tools that perform write operations (delete data, send emails, deploy code), implement a gatekeeper:

```python
class GatekeeperTool:
    def __init__(self, tool, requires_confirmation=True):
        self.tool = tool
        self.requires_confirmation = requires_confirmation
    
    def execute(self, **kwargs):
        if self.requires_confirmation:
            # Log the intended action for human review
            approval = request_human_approval(
                tool=self.tool.name,
                action=kwargs,
                reason="Write operation requires confirmation"
            )
            if not approval.granted:
                return "Action blocked by gatekeeper"
        
        return self.tool.execute(**kwargs)
```

---

## 7.4 MCP (Model Context Protocol)

### What It Is

MCP is an open standard (initiated by Anthropic) that defines how LLM applications discover, register, and invoke tools. Think of it as "USB-C for AI tools" — a universal interface that any model can use with any tool server.

### Architecture: Host → Client → Server

```
┌─────────────────────────────────────────────────┐
│  Host (e.g., Claude Desktop, IDE, Custom App)   │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Client 1 │  │ Client 2 │  │ Client 3 │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
└───────┼──────────────┼──────────────┼───────────┘
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Server  │   │ Server  │   │ Server  │
   │ (Git)   │   │ (DB)    │   │ (API)   │
   └─────────┘   └─────────┘   └─────────┘
```

- **Host:** The application that embeds the LLM (provides the UI, manages conversations)
- **Client:** Maintains a 1:1 connection with a single MCP server
- **Server:** Exposes tools, resources, and prompts via the MCP protocol

### JSON-RPC 2.0: The Wire Format

All MCP communication uses JSON-RPC 2.0:

**Request (Client → Server):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_signal_dependencies",
    "arguments": {
      "signal_name": "SIG_BRK_PRESSURE_FL",
      "direction": "upstream"
    }
  }
}
```

**Response (Server → Client):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 3 upstream dependencies: [SIG_WHEEL_SPD, SIG_ABS_CTRL, SIG_ESP_REQ]"
      }
    ]
  }
}
```

### Tool Registration

When an MCP server starts, the client calls `tools/list` to discover available tools:

```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "method": "tools/list"
}
```

The server responds with all tool schemas:

```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "result": {
    "tools": [
      {
        "name": "get_signal_dependencies",
        "description": "Returns signal dependencies from the graph",
        "inputSchema": {
          "type": "object",
          "properties": { ... },
          "required": ["signal_name"]
        }
      }
    ]
  }
}
```

### Why MCP Matters

1. **Interoperability:** Same tools work with Claude, GPT, Gemini, local models
2. **Separation of concerns:** Tool logic decoupled from LLM application
3. **Ecosystem:** Community-built servers for databases, APIs, file systems
4. **Extensibility:** Add new capabilities without modifying the agent
5. **Security:** Servers run in isolated processes with defined permissions

---

## 7.5 Agent-to-Agent (A2A) Communication

### Why Multi-Agent?

Complex systems benefit from specialization. Rather than one monolithic agent that knows everything, you decompose into focused agents:

- **Research Agent:** Searches the web, reads documents
- **Code Agent:** Writes and reviews code
- **Planning Agent:** Breaks goals into sub-tasks
- **Critic Agent:** Evaluates quality and catches errors

### Communication Patterns

**Orchestrator/Mediator:**
```
         ┌─────────────┐
         │ Orchestrator │
         └──────┬──────┘
        ┌───────┼───────┐
        ▼       ▼       ▼
   ┌────────┐ ┌────┐ ┌──────┐
   │Research│ │Code│ │Review│
   └────────┘ └────┘ └──────┘
```
One central agent delegates tasks and aggregates results. Simple to reason about, but the orchestrator is a bottleneck.

**Peer-to-Peer:**
```
   ┌────────┐ ←──→ ┌────┐
   │Research│       │Code│
   └────────┘ ←──→ └────┘
        ↕              ↕
   ┌──────────────────────┐
   │       Review         │
   └──────────────────────┘
```
Agents communicate directly. More flexible, but harder to debug.

**Hierarchical:**
```
         ┌──────────┐
         │ Manager  │
         └────┬─────┘
       ┌──────┼──────┐
       ▼      ▼      ▼
   ┌──────┐┌──────┐┌──────┐
   │Lead 1││Lead 2││Lead 3│
   └──┬───┘└──┬───┘└──┬───┘
      ▼       ▼       ▼
   Workers  Workers  Workers
```
Tree structure with delegation at each level. Good for large-scale systems.

### JSON-RPC 2.0 Between Agents

Agents can communicate using the same JSON-RPC 2.0 format as MCP:

```json
{
  "jsonrpc": "2.0",
  "id": "task-42",
  "method": "investigate",
  "params": {
    "signal": "SIG_BRK_PRESSURE_FL",
    "priority": "high",
    "deadline_seconds": 60
  }
}
```

### Task Delegation and Result Aggregation

```python
class OrchestratorAgent:
    def delegate_investigation(self, signal_failure):
        # Fan out to specialized agents
        tasks = [
            self.graph_agent.trace_dependencies(signal_failure.signal),
            self.git_agent.find_recent_changes(signal_failure.module),
            self.config_agent.check_configuration(signal_failure.variant),
        ]
        
        # Gather results
        results = await asyncio.gather(*tasks)
        
        # Synthesize
        return self.synthesize(results, signal_failure)
```

---

## 7.6 LangGraph: Building Agent Workflows

LangGraph (from LangChain) provides a framework for building agent workflows as directed graphs with typed state.

### StateGraph: The Foundation

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    query_expanded: str
    documents: list[str]
    answer: str

# Define the graph
graph = StateGraph(AgentState)
```

### Nodes and Edges

Each node is a function that takes state and returns updated state:

```python
def classify_intent(state: AgentState) -> dict:
    """Determine what the user wants."""
    messages = state["messages"]
    intent = llm.invoke(
        f"Classify this query: {messages[-1].content}"
    )
    return {"intent": intent.content}

def expand_query(state: AgentState) -> dict:
    """Rewrite the query for better retrieval."""
    expanded = llm.invoke(
        f"Expand this search query: {state['messages'][-1].content}"
    )
    return {"query_expanded": expanded.content}

def retrieve_documents(state: AgentState) -> dict:
    """Fetch relevant documents from vector store."""
    docs = vector_store.similarity_search(state["query_expanded"], k=5)
    return {"documents": [d.page_content for d in docs]}

def generate_answer(state: AgentState) -> dict:
    """Generate final answer from retrieved context."""
    context = "\n".join(state["documents"])
    answer = llm.invoke(
        f"Answer based on context:\n{context}\n\nQuestion: {state['messages'][-1].content}"
    )
    return {"answer": answer.content}

# Add nodes
graph.add_node("classify", classify_intent)
graph.add_node("expand", expand_query)
graph.add_node("retrieve", retrieve_documents)
graph.add_node("generate", generate_answer)
```

### Conditional Edges

Route execution based on state:

```python
def route_by_intent(state: AgentState) -> str:
    if state["intent"] == "factual":
        return "expand"       # needs retrieval
    elif state["intent"] == "conversational":
        return "generate"     # direct response
    else:
        return "expand"       # default to retrieval

graph.add_conditional_edges("classify", route_by_intent)
graph.add_edge("expand", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

graph.set_entry_point("classify")
```

### Checkpointing: MemorySaver

Persist state across invocations for multi-turn conversations:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# First invocation
config = {"configurable": {"thread_id": "user-session-123"}}
result = app.invoke({"messages": [HumanMessage("What is iFAST?")]}, config)

# Second invocation — remembers the first
result = app.invoke({"messages": [HumanMessage("How does it work?")]}, config)
```

---

## 7.7 State Management & Memory

### Session State: What to Persist

Not everything belongs in memory. Persist:
- Conversation history (messages)
- User preferences discovered during interaction
- Tool results that might be referenced again
- Current task progress (for recovery)

Do not persist:
- Internal reasoning traces (too verbose)
- Intermediate tool outputs that are already summarized
- Temporary variables used in a single step

### Short-Term Memory (Conversation History)

The simplest form: keep the last $N$ messages in context.

```python
class SlidingWindowMemory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages = []
    
    def add(self, message):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            # Keep system message + trim oldest
            self.messages = [self.messages[0]] + self.messages[-self.max_messages:]
    
    def get_context(self) -> list:
        return self.messages
```

### Long-Term Memory

For information that spans sessions:

**Vector Store Memory:**
```python
def store_memory(conversation_summary: str, metadata: dict):
    embedding = embed_model.encode(conversation_summary)
    vector_store.upsert(
        id=metadata["session_id"],
        vector=embedding,
        metadata=metadata,
        text=conversation_summary
    )

def recall_relevant(query: str, k: int = 5) -> list[str]:
    results = vector_store.similarity_search(query, k=k)
    return [r.text for r in results]
```

**Summary Memory:** Periodically summarize old messages to compress context:

```python
def compress_history(messages: list, max_tokens: int = 2000):
    if count_tokens(messages) < max_tokens:
        return messages
    
    # Summarize older messages
    old_messages = messages[1:-5]  # Keep system + last 5
    summary = llm.invoke(f"Summarize this conversation:\n{old_messages}")
    
    return [messages[0], SystemMessage(f"Previous context: {summary}")] + messages[-5:]
```

### Checkpointing for Fault Tolerance

If an agent crashes mid-investigation, checkpointing lets it resume:

```python
class CheckpointedAgent:
    def __init__(self, checkpoint_store):
        self.store = checkpoint_store
    
    def run(self, task_id: str, goal: str):
        # Try to resume from checkpoint
        checkpoint = self.store.get(task_id)
        if checkpoint:
            state = checkpoint.state
            step = checkpoint.step
        else:
            state = initial_state(goal)
            step = 0
        
        for i in range(step, self.max_steps):
            state = self.execute_step(state)
            self.store.save(task_id, state, step=i)
            
            if state.is_complete:
                return state.result
```

---

## 7.8 Error Recovery & Resilience

Production agents must handle failures gracefully. The real world is messy.

### Retry with Exponential Backoff

Using the `tenacity` library:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
)
def call_llm(messages, tools):
    return bedrock_client.invoke_model(
        modelId="us.anthropic.claude-sonnet-4-20250514",
        messages=messages,
        tools=tools
    )
```

### Fallback Models

When the primary model is unavailable or rate-limited:

```python
class ModelWithFallback:
    def __init__(self, primary: str, fallback: str):
        self.primary = primary
        self.fallback = fallback
    
    def invoke(self, messages, tools):
        try:
            return call_model(self.primary, messages, tools)
        except (RateLimitError, ServiceUnavailableError):
            logger.warning(f"Primary model {self.primary} failed, using fallback")
            return call_model(self.fallback, messages, tools)

# Usage
model = ModelWithFallback(
    primary="us.anthropic.claude-sonnet-4-20250514",
    fallback="us.anthropic.claude-haiku-3-20250301"
)
```

### Graceful Degradation

Partial results are better than no results:

```python
def investigate_signal(signal: str) -> InvestigationResult:
    results = {}
    
    # Each source is independent — don't let one failure kill the whole investigation
    for source in evidence_sources:
        try:
            results[source.name] = source.gather(signal)
        except Exception as e:
            logger.error(f"Source {source.name} failed: {e}")
            results[source.name] = EvidenceResult(
                available=False,
                error=str(e)
            )
    
    # Proceed with whatever evidence we collected
    available_results = {k: v for k, v in results.items() if v.available}
    if len(available_results) >= 3:  # minimum viable evidence
        return fuse_evidence(available_results)
    else:
        return InvestigationResult(status="INSUFFICIENT_EVIDENCE")
```

### Timeout Handling

Agents must respect time budgets:

```python
import asyncio

async def investigate_with_timeout(signal: str, timeout_seconds: int = 60):
    try:
        result = await asyncio.wait_for(
            agent.investigate(signal),
            timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        # Return best partial result
        return agent.get_partial_result()
```

---

## 7.9 Case Study: iFAST — Intelligent Failure Analysis through Signal Tracing

### The Problem

At Mercedes-Benz R&D, automotive software is validated through continuous integration testing. Thousands of signals (brake pressure, wheel speed, engine torque) flow between hundreds of software modules. When a test fails — a signal produces unexpected values — an engineer must investigate.

**The manual process:**
1. Identify which signal failed
2. Trace the signal's dependencies through the architecture
3. Check if any upstream module's source code changed
4. Compare binary outputs to previous baselines
5. Examine configuration files for variant mismatches
6. Review git history for relevant commits
7. Cross-reference with known issues
8. Write a root-cause report

**Time per investigation:** 4-8 hours for a senior engineer.

**Scale of the problem:** 50+ new failures per week. The backlog grew 10x faster than the team could handle. Engineers were spending 80% of their time on investigation, leaving 20% for actual development.

### The Solution

**iFAST** (Intelligent Failure Analysis through Signal Tracing): an autonomous AI agent that performs the entire investigation in under 60 seconds, achieving 97% accuracy on root-cause classification.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    iFAST Agent System                         │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Claude Agent (AWS Bedrock)                │       │
│  │         ReAct Loop with 14 MCP Tools             │       │
│  └──────────────────┬───────────────────────────────┘       │
│                     │ JSON-RPC 2.0                           │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────┐       │
│  │              MCP Tool Server                      │       │
│  │                                                   │       │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐          │       │
│  │  │  Graph  │ │  Code   │ │   Git    │          │       │
│  │  │Traversal│ │Analysis │ │  History │          │       │
│  │  └─────────┘ └─────────┘ └──────────┘          │       │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐          │       │
│  │  │ Config  │ │ Visual  │ │  Binary  │          │       │
│  │  │  Check  │ │  Diff   │ │  Compare │          │       │
│  │  └─────────┘ └─────────┘ └──────────┘          │       │
│  │  ┌─────────┐ ┌─────────┐                        │       │
│  │  │ Report  │ │Progress │                        │       │
│  │  │  Gen    │ │ Stream  │                        │       │
│  │  └─────────┘ └─────────┘                        │       │
│  └──────────────────────────────────────────────────┘       │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────┐       │
│  │        Bayesian Evidence Fusion Engine            │       │
│  │        8 Sources | Calibrated LLR Weights        │       │
│  └──────────────────────────────────────────────────┘       │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────┐       │
│  │       Source-Code Dependency Graph                │       │
│  │       1.6M+ edges | 1,128 modules               │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**Key components:**
- **Single Claude agent** on AWS Bedrock — one coherent reasoning chain
- **14 MCP tools** — registered via JSON-RPC 2.0, called as needed
- **Bayesian evidence fusion** — 8 independent evidence sources, calibrated Log-Likelihood Ratios
- **Dependency graph** — 1.6M+ edges parsed from 1,128 software modules
- **Real-time progress streaming** — file-based IPC for non-blocking UI updates

### The 14 Tools

The agent has access to 14 specialized tools, grouped by function:

| Category | Tools | Purpose |
|----------|-------|---------|
| Graph Traversal | `trace_upstream`, `trace_downstream`, `find_shortest_path` | Navigate the 1.6M-edge dependency graph |
| Code Analysis | `analyze_source_changes`, `check_signal_definition`, `search_codebase` | Examine source code for the signal |
| Git History | `get_commit_diff`, `get_module_changes`, `blame_line` | Track what changed and when |
| Configuration | `check_variant_config`, `compare_build_settings` | Detect configuration mismatches |
| Visual Diff | `visual_structural_diff`, `compare_signal_diagrams` | Compare architectural diagrams |
| Binary Comparison | `compare_binary_outputs` | Diff compiled module outputs |
| Reporting | `generate_report`, `stream_progress` | Output results and status |

The agent does not use all 14 tools on every investigation. The ReAct loop lets it decide which tools are relevant based on the specific failure.

### Bayesian Evidence Fusion

The core innovation: instead of a heuristic decision tree, iFAST uses **Bayesian evidence fusion** to combine independent evidence sources into a calibrated confidence score.

**The mathematical framework:**

Each evidence source $i$ contributes a **Log-Likelihood Ratio (LLR)**:

$$LLR_i = \log \frac{P(Evidence_i | CodeChange)}{P(Evidence_i | \neg CodeChange)}$$

The posterior probability is computed by summing all LLRs:

$$P(CodeChange | AllEvidence) = \sigma\left(\sum_{i=1}^{N} LLR_i\right)$$

where $\sigma$ is the sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$.

**Calibrated LLR weights** (learned from historical investigations):

| Evidence Source | LLR Weight | Meaning |
|----------------|-----------|---------|
| `visual_structural_diff` (structure changed) | +3.5 | Strong evidence of code change |
| `source_code_deleted` | +3.0 | Signal definition removed |
| `git_module_changed` (recent commit) | +2.5 | Module was recently modified |
| `binary_output_differs` | +2.0 | Compiled output changed |
| `config_variant_mismatch` | +1.5 | Build configuration different |
| `upstream_dependency_changed` | +1.5 | Upstream module affected |
| `signal_UNCHANGED` (still in code) | -2.5 | Counter-evidence: signal exists |
| `no_git_changes` (module untouched) | -2.0 | Counter-evidence: nothing changed |

**Cross-validation logic:**

```python
def classify_confidence(evidence_results: dict) -> str:
    total_llr = sum(e.llr for e in evidence_results.values() if e.available)
    
    # Count agreeing sources
    positive_sources = [e for e in evidence_results.values() if e.llr > 0]
    
    if total_llr < 0:
        return "INCONCLUSIVE"
    elif len(positive_sources) >= 3:
        return "TRIPLE_CONFIRMED"
    elif len(positive_sources) >= 2:
        return "DOUBLE_CONFIRMED"
    else:
        return "SINGLE_SOURCE"
```

**Threshold rule:** If $\sum LLR_i < 0$, the result is **INCONCLUSIVE** — the agent will not make a false positive claim. This is critical in automotive contexts where incorrect diagnoses waste engineering time.

**Example investigation:**

```
Signal: SIG_BRK_PRESSURE_FL
Status: FAILED (value=0, expected=non-zero)

Evidence gathered:
  visual_structural_diff:  +3.5  (diagram shows removed connection)
  source_code_deleted:     +3.0  (signal publish call deleted)
  git_module_changed:      +2.5  (commit abc123, 2 days ago)
  signal_UNCHANGED:        N/A   (signal NOT found — confirms deletion)
  
Total LLR = 3.5 + 3.0 + 2.5 = 9.0
P(CodeChange) = sigmoid(9.0) = 0.9999

Classification: TRIPLE_CONFIRMED
Root Cause: Signal SIG_BRK_PRESSURE_FL was deleted in commit abc123
            by developer X in module BrakeControl.
```

### Real-Time Progress Streaming

Investigations take ~60 seconds. Users need feedback, not a blank screen.

**Solution:** File-based IPC with non-blocking reads:

```python
class ProgressStreamer:
    def __init__(self, session_id: str):
        self.progress_file = f"/tmp/ifast_{session_id}_progress.json"
    
    def update(self, step: str, percent: int, detail: str):
        progress = {
            "step": step,
            "percent": percent,
            "detail": detail,
            "timestamp": time.time()
        }
        # Atomic write to prevent partial reads
        tmp_file = self.progress_file + ".tmp"
        with open(tmp_file, 'w') as f:
            json.dump(progress, f)
        os.replace(tmp_file, self.progress_file)
```

The UI polls this file every 500ms to display:
```
[Step 3/7] Analyzing git history... (42%)
  Found 3 commits affecting BrakeControl module
```

### Results

| Metric | Before iFAST | After iFAST | Improvement |
|--------|-------------|-------------|-------------|
| Time per investigation | 4-8 hours | 60 seconds | **54.8% time savings** (accounting for review) |
| Accuracy | N/A (human baseline) | 97% | — |
| False positive rate | ~30% (initial heuristic) | ~7.5% | **75% reduction** |
| Weekly capacity | ~10 investigations | 50+ investigations | **5x throughput** |
| Teams using system | 0 | 2+ | Growing adoption |

The "54.8% time savings" accounts for the fact that engineers still review the agent's output — the investigation is automated, but the final decision remains human. Total end-to-end time dropped from 4-8 hours to ~30 minutes (agent runs 60s + engineer reviews 25 minutes).

### Key Engineering Decisions

**1. Why single agent (not multi-agent)?**

Root-cause analysis requires a coherent chain of reasoning. The agent needs to see how evidence from one tool informs its choice of the next tool. With multiple agents, you lose this coherence — the graph-traversal agent doesn't know what the git agent found.

A single agent with many tools outperforms a multi-agent system for investigation tasks because:
- Evidence is correlated (git changes AND binary diffs relate to the same commit)
- The reasoning chain must be unbroken (finding A leads to checking B)
- Coordination overhead between agents would exceed the 60-second budget

**2. Why Bayesian (not heuristic)?**

The previous system used a decision tree: "if git changed AND binary different → code change." This produced 30% false positives.

Bayesian fusion provides:
- **Mathematical grounding:** Each weight has a probabilistic interpretation
- **Calibration:** Weights are tuned on historical data, not gut feelings
- **Interpretability:** You can explain *why* the system concluded "code change" by listing which evidence contributed and how much
- **Graceful handling of missing evidence:** If a source is unavailable, it simply doesn't contribute (LLR = 0), rather than breaking a decision tree

**3. Why MCP (not hardcoded tool calls)?**

The system started with 8 tools. Within 3 months, it grew to 14. MCP made this painless:
- New tools are added by implementing an MCP server — no agent code changes
- Tool schemas are auto-discovered at startup
- The agent's system prompt doesn't need updating (tool descriptions come from MCP)
- Different deployments can have different tool sets (some teams don't need binary comparison)

**4. Why async progress (not blocking)?**

User research showed that engineers abandoned the tool if they saw no feedback for >10 seconds. They assumed it was broken. File-based IPC was chosen over WebSockets because:
- No additional infrastructure needed (no message broker)
- Works in air-gapped environments (automotive networks)
- Trivial to debug (just `cat` the progress file)
- No connection management complexity

---

## 7.10 Summary & Interview Tips

### Key Points to Remember

1. **Agents = Perception + Reasoning + Action in a loop.** The loop is what differentiates agents from chatbots.

2. **ReAct is the workhorse pattern.** Think → Act → Observe → Repeat. Simple, effective, widely applicable.

3. **Tools are JSON Schema defined.** Name, description (critical for selection), parameters. The LLM generates structured calls; the runtime executes them.

4. **MCP standardizes tool interfaces.** Host → Client → Server. JSON-RPC 2.0. Enables interoperability.

5. **State management is the hard problem.** What to keep, what to summarize, what to discard. Checkpointing enables fault tolerance.

6. **Error recovery is non-negotiable in production.** Retries, fallbacks, graceful degradation, timeouts. Plan for failure.

7. **Bayesian evidence fusion > heuristics** when you have multiple independent signals. Calibrated, interpretable, mathematically sound.

8. **Single-agent with many tools** often beats multi-agent for investigation tasks. Coherent reasoning chain matters.

### Common Interview Questions

**Q: "What is the difference between an AI agent and a chatbot?"**

A: A chatbot is stateless and reactive — it answers the current question with no memory or tool use. An agent is stateful, tool-using, and goal-directed. It maintains memory across turns, can call external tools to gather information or take actions, and works toward achieving a multi-step objective autonomously.

**Q: "When would you use multi-agent vs. single agent?"**

A: Use multi-agent when tasks are truly independent and parallelizable (e.g., one agent researches while another writes code). Use single agent when the reasoning chain must be coherent — when the output of one step directly informs the next. In iFAST, we chose single agent because root-cause analysis requires seeing all evidence together.

**Q: "How do you handle tool failures in an agent system?"**

A: Three layers: (1) Retry with exponential backoff for transient failures (network, rate limits). (2) Fallback alternatives (different model, different data source). (3) Graceful degradation — proceed with partial results rather than failing entirely. The agent sees the error in its observation and can reason about alternatives.

**Q: "Explain the ReAct pattern with an example."**

A: ReAct alternates reasoning and action. Example: "Thought: I need to find why the test failed. Let me check the git log. Action: git_log(module='brake'). Observation: Commit abc123 deleted signal publication. Thought: The signal was deleted. Let me verify it's not moved elsewhere. Action: search('SIG_BRK'). Observation: Not found. Thought: Confirmed deletion. Root cause identified."

**Q: "What is MCP and why does it matter?"**

A: MCP (Model Context Protocol) is an open standard for how LLM applications discover and invoke tools. It uses JSON-RPC 2.0 with a Host → Client → Server architecture. It matters because it decouples tool implementation from the LLM application — the same tools work with any model provider, and new tools can be added without modifying the agent.

**Q: "How do you evaluate agent performance?"**

A: Multiple dimensions: (1) Task completion rate — did it achieve the goal? (2) Accuracy — are the results correct? (3) Efficiency — how many steps/tokens/seconds? (4) Robustness — how does it handle edge cases and failures? (5) User trust — do users accept and act on its outputs? For iFAST, we measured accuracy (97%), time savings (54.8%), and false positive reduction (75%).

**Q: "How do you prevent agents from hallucinating or making incorrect tool calls?"**

A: (1) Grounding — require the agent to cite evidence from tool outputs, not generate facts. (2) Structured outputs — use JSON Schema to constrain tool call parameters. (3) Verification loops — have the agent check its own work (Reflexion pattern). (4) Confidence thresholds — if evidence is insufficient (LLR < 0), return INCONCLUSIVE rather than guessing. (5) Human-in-the-loop for high-stakes actions.

---

*Next chapter: Chapter 8 — Retrieval-Augmented Generation (RAG)*
