# Chapter 10: Search Algorithms for LLM Reasoning

## 10.1 Why Search Over Reasoning Traces?

Large Language Models generate text left-to-right, one token at a time. Chain-of-Thought (CoT) prompting asks the model to "think step by step" — but this produces a single linear chain of reasoning. If the model makes a wrong turn at step 3, every subsequent step builds on that error.

**The core insight:** What if we could explore *multiple* reasoning paths simultaneously, just like a chess engine explores multiple move sequences?

This is exactly what A* search over reasoning traces does. Instead of generating one reasoning chain, we treat each intermediate reasoning state as a node in a graph and explore multiple continuations, prioritized by a heuristic estimate of how close each path is to a correct answer.

**When does this help?**
- Multi-hop questions (need 2+ reasoning steps that must connect)
- Problems where the model "knows" the answer but can't find it in one pass
- When different reasoning strategies lead to different answers (complementarity)

**When does CoT suffice?**
- Single-hop factual questions
- Short, unambiguous problems
- When the model is highly confident (low entropy)

---

## 10.2 A* Algorithm: The Fundamentals

### Definition

A* is a best-first graph search algorithm that finds the least-cost path from a start node to a goal node. It uses an evaluation function:

$$f(n) = g(n) + h(n)$$

Where:
- $g(n)$ = actual cost from start to node $n$ (known exactly)
- $h(n)$ = heuristic estimate of cost from $n$ to goal (estimated)
- $f(n)$ = estimated total cost of the cheapest solution through $n$

### Algorithm (Pseudocode)

```python
def astar_search(start, is_goal, expand, heuristic):
    open_set = MinHeap()  # Priority queue on f_score
    open_set.push(start, f=0 + heuristic(start))
    visited = set()
    
    while open_set:
        current = open_set.pop()  # Lowest f_score
        
        if is_goal(current):
            return current  # Found optimal path
        
        if current.signature() in visited:
            continue
        visited.add(current.signature())
        
        for child, step_cost in expand(current):
            child.g_score = current.g_score + step_cost
            child.h_score = heuristic(child)
            child.f_score = child.g_score + child.h_score
            open_set.push(child)
    
    return None  # No solution found
```

### Key Properties

| Property | Requirement | Guarantees |
|----------|-------------|------------|
| **Completeness** | Finite branching factor, positive step costs | Will find a solution if one exists |
| **Optimality** | Admissible heuristic | Finds the *lowest-cost* solution |
| **Optimality (graph search)** | Consistent heuristic | Same guarantee with visited set |

---

## 10.3 Admissibility and Consistency

### Admissibility

A heuristic $h(n)$ is **admissible** if it never overestimates the true cost to reach the goal:

$$h(n) \leq h^*(n) \quad \forall n$$

Where $h^*(n)$ is the true minimum cost from $n$ to the nearest goal.

**Why it matters:** If the heuristic overestimates, A* might skip the optimal path (it looks too expensive) and return a suboptimal solution.

### Consistency (Monotonicity)

A heuristic is **consistent** if for every node $n$ and every successor $n'$:

$$h(n) \leq c(n, n') + h(n')$$

Where $c(n, n')$ is the cost of the edge from $n$ to $n'$.

**Why it matters:** Consistency implies admissibility, AND it guarantees that once a node is expanded (popped from the open set), its $g$-score is already optimal. This allows us to use a `visited` set (graph search) without losing optimality.

**Intuition:** The heuristic estimate can never "jump up" by more than the actual step cost. It decreases smoothly as you get closer to the goal.

---

## 10.4 Heuristic Design for LLM Reasoning

### Entity-Coverage Heuristic (HotpotQA, StrategyQA)

**Idea:** A good reasoning trace should address all entities mentioned in the question. If the question asks about "Albert Einstein" and "Princeton University" but the trace only mentions Einstein, it's probably not done reasoning.

```python
WEIGHT_DEPTH    = 0.3  # Per remaining hop
WEIGHT_COVERAGE = 0.3  # For entity coverage

def compute_heuristic(state, question_entities):
    # Component 1: How many more hops are needed?
    h_depth = max(0, 2 - state.depth) * WEIGHT_DEPTH
    
    # Component 2: What fraction of question entities are unaddressed?
    coverage = entity_coverage(state, question_entities)
    h_coverage = (1.0 - coverage) * WEIGHT_COVERAGE
    
    return h_depth + h_coverage
```

**Entity extraction:** Quoted strings, capitalized phrases (proper nouns), numbers/years, content words (length > 3, excluding stopwords).

**Coverage computation:**
$$\text{coverage}(s) = \frac{|\text{question\_entities} \cap \text{trace\_entities}(s)|}{|\text{question\_entities}|}$$

### Logic-Complexity Heuristic (LogicQA)

For multiple-choice logical reasoning, entity coverage doesn't make sense (there are no "entities to cover"). Instead, we estimate remaining difficulty from the logical structure:

```python
LAMBDA = 0.12

def compute_heuristic(state, context, query, max_depth=8):
    h1 = LAMBDA * max(0, max_depth - state.depth)  # Remaining budget
    
    # Logic complexity analysis
    complexity = 2  # Base
    if connective_count >= 3: complexity += 1  # if/then/unless/...
    if connective_count >= 6: complexity += 1
    if has_hard_type: complexity += 1  # weaken/strengthen/assumption
    if sentence_count >= 5: complexity += 1
    if sentence_count >= 10: complexity += 1
    
    h2 = LAMBDA * max(0, complexity - state.depth)
    return (h1 + h2) / 2
```

### Depth-Only Heuristic (Code Benchmarks)

For code generation, the heuristic is simpler — we estimate how many planning steps remain before we can generate code:

```python
def compute_code_heuristic(node, problem):
    if node.code:
        return 0.0  # Goal state
    return max(0, 3 - node.depth) * 0.3
```

---

## 10.5 Admissibility Proof (HotpotQA / StrategyQA)

This is the formal proof from the paper. You should be able to reproduce this on a whiteboard.

**Given:**
- Step cost: $c(s \to s') = 1.0 + (1.0 - \text{confidence}) \geq 1.01$ (confidence capped at 0.99)
- Minimum hops to goal: 2 (HotpotQA is multi-hop by design)
- Minimum total path cost: $h^*(\text{root}) \geq 2 \times 1.01 = 2.02$

**Heuristic at each depth:**
- $d = 0$: $h \leq 2 \times 0.3 + 1 \times 0.3 = 0.9$
- $d = 1$: $h \leq 1 \times 0.3 + 1 \times 0.3 = 0.6$
- $d \geq 2$: $h \leq 0 \times 0.3 + 1 \times 0.3 = 0.3$

**Admissibility check:**
- $d = 0$: $0.9 < 2.02 = h^*(\text{root})$ ✓
- $d = 1$: $0.6 < 1.01 = h^*(\text{at depth 1})$ ✓
- $d \geq 2$: $0.3 < 1.01$ ✓

**Consistency check:**
$$\max(\Delta h) = |h(n) - h(n')| \leq 0.3 + 0.3 = 0.6 < 1.01 = \min(c)$$

Since $\Delta h < c_{\min}$ for all transitions, the heuristic is consistent.

**Conclusion:** A* with graph search is **optimal and complete** — guaranteed to find the lowest-cost reasoning path. □

---

## 10.6 Cost Functions

The cost function determines how expensive each reasoning step is:

$$c(s \to s') = 1.0 + (1.0 - \text{confidence})$$

Where confidence is the LLM's self-reported certainty (parsed from structured output), clamped to [0.01, 0.99].

| Confidence | Step Cost | Interpretation |
|------------|-----------|----------------|
| 0.99 (max) | 1.01 | Very confident step — cheap to take |
| 0.80 | 1.20 | Normal step |
| 0.50 | 1.50 | Uncertain step — expensive |
| 0.01 (min) | 1.99 | Very uncertain — almost 2× base cost |

**Why this design:**
- Base cost of 1.0 ensures all steps are positive (required for A* correctness)
- Uncertainty penalty discourages exploring paths through low-confidence reasoning
- Clamping at 0.99 ensures $c_{\min} = 1.01 > 0$ (needed for admissibility proof)

### Code Benchmark Cost (Asymmetric)

For code generation, planning steps are more expensive than final code generation:

| Phase | Step Cost | Rationale |
|-------|-----------|-----------|
| Planning (depth < 2) | 1.3 | Uncertainty in reasoning direction |
| Code generation (depth ≥ 2) | 0.5 | Plan exists; execution is cheaper |

This encourages the search to move quickly to code generation once a plan is formed.

---

## 10.7 Branching Strategies

### The Problem: Ollama Returns Only 1 Response

Standard LLM APIs support `n > 1` to get multiple completions. But Ollama's chat completion endpoint **ignores** this parameter and always returns 1 response. This makes naive A* degenerate into a linear chain (no actual search — just CoT with extra steps).

### The Solution: K Separate API Calls at Varied Temperatures

```python
BRANCH_K = 3
TEMPERATURES = [0.15, 0.45, 0.75]

for i in range(BRANCH_K):
    response = llm.chat(
        messages=prompt,
        temperature=TEMPERATURES[i]
    )
    # Each temperature produces genuinely different continuation
    children.append(parse_response(response))
```

**Why varied temperatures work:**
- T=0.15: Precise, conservative reasoning (likely the "textbook" answer)
- T=0.45: Balanced — explores some alternatives
- T=0.75: Creative — considers unusual connections

This produces true diversity in the search tree. Without it, all branches would be near-identical (defeating the purpose of search).

### Branching Factor by Dataset

| Dataset | K | MAX_NODES | Rationale |
|---------|---|-----------|-----------|
| HotpotQA | 3 | 30 | Multi-hop needs diverse paths |
| StrategyQA | 2 | 15 | Yes/no with world knowledge — less branching needed |
| LogicQA | 2 | 12 | MCQ — fewer valid reasoning paths |
| Code benchmarks | 1 | 15 | Planning → execution is more linear |

---

## 10.8 Goal Detection and Self-Consistency Voting

### Goal Detection

A node is a **goal** when it contains a final answer:
- QA: regex extracts "Final Answer: X" or "The answer is: X"
- Code: `node.code != ""` (complete code generated)

### Self-Consistency Voting

**Key insight:** Don't stop at the first goal found. Continue searching until budget exhausted, collect ALL goal nodes, then vote.

```python
goals = []
while open_set and nodes_explored < MAX_NODES:
    current = heapq.heappop(open_set)
    
    if is_goal(current):
        goals.append(current)
        if majority_agrees(goals, threshold=MAJORITY_VOTE):
            break  # Early exit: confident answer
        continue  # Keep searching for more goals
    
    # ... expand current node

# Final answer: majority vote across all goals
answer = majority_vote(goals)
```

**Why this works:** Different reasoning paths may reach the same correct answer through different routes. If 3 out of 4 goal nodes agree on "yes," that's much more reliable than a single path's answer.

**Weight by cost:** Goals reached via lower-cost paths (more confident reasoning) get higher weight in the vote.

---

## 10.9 Budget-Bounded Search

Real LLM calls are expensive. We bound search with two parameters:

- **MAX_NODES:** Maximum nodes to expand (= maximum LLM calls)
- **MAX_DEPTH:** Maximum reasoning chain length

| Parameter | Effect of Too Low | Effect of Too High |
|-----------|-------------------|-------------------|
| MAX_NODES | Misses good paths | Wastes compute |
| MAX_DEPTH | Can't solve deep problems | Long, unfocused chains |

**Finding the sweet spot:** Our experiments show that most correct answers are found within 8-15 nodes. Beyond 30 nodes, the search rarely finds new answers (diminishing returns).

---

## 10.10 Deduplication and Cycle Detection

Without deduplication, A* can revisit equivalent states (wasting budget):

```python
def signature(state):
    """Unique identity of a reasoning state."""
    return " || ".join(step.content.strip() for step in state.steps)

visited = set()
# In main loop:
sig = current.state.signature()
if sig in visited:
    continue  # Skip — already explored this state
visited.add(sig)
```

Additionally, **degenerate output detection** filters LLM responses that are repetitive garbage:

```python
def is_degenerate(content):
    if len(set(content.strip())) <= 3 and len(content) > 20:
        return True  # Same character repeated
    # Word frequency analysis for loops
    words = content.split()
    if len(words) > 10:
        freq = Counter(words)
        if freq.most_common(1)[0][1] / len(words) > 0.5:
            return True  # One word dominates
    return False
```

---

## 10.11 Case Study: MTP Thesis A* Implementation

### System Overview

Evaluated on 8 benchmarks × 3 model scales × 4 methods = 96 configurations, 500 instances each (48,000+ total evaluations).

**Models:** Qwen-2.5-0.5B, LLaMA-3.1-8B, Qwen-2.5-14B  
**Benchmarks:** StrategyQA, LogicQA, HotpotQA, GSM8K, MATH, HumanEval, MBPP, CodeContests  
**Methods:** Chain-of-Thought, A*, Self-Consistency (k=5), Tree-of-Thought (k=3)

### Key Results

| Benchmark | CoT (8B) | A* (8B) | Δ | When A* Wins |
|-----------|----------|---------|---|---|
| HotpotQA | 68.0% | 70.1% | +2.1 | Multi-hop bridge questions |
| CodeContests | 77.4% | 87.0% | +9.6 | Complex algorithmic problems |
| StrategyQA | 65.0% | 75.0% | +10.0 | World-knowledge chains |

### Architecture Decisions

1. **Why A* (not BFS/DFS)?** A* with admissible heuristic guarantees optimality. BFS finds shallowest solution (not lowest-cost). DFS can get stuck in dead ends.

2. **Why entity-coverage heuristic?** It's informative (distinguishes good paths from bad at the same depth) AND admissible (never overestimates). Depth-only heuristic gives h=0 for all nodes at depth ≥ 2 — no guidance.

3. **Why self-consistency voting?** A single goal node can be wrong (the model "got lucky" on one path). Multiple goals provide statistical reliability.

4. **Why budget-bounded?** Unbounded A* on an LLM could generate infinite nodes (the LLM always has something to say). Fixed budget makes runtime predictable.

---

## 10.12 Comparison with Other Search Methods

| Method | Guarantees | LLM Calls | Key Weakness |
|--------|-----------|-----------|--------------|
| **CoT** | None | 1 | Single path — no recovery from errors |
| **Self-Consistency** | Statistical (majority vote) | k (typically 5) | No directed search — all paths are independent |
| **Tree-of-Thought** | None (greedy selection) | O(k × d) | Evaluation step can be wrong; prunes good paths |
| **A*** | Optimal (with admissible h) | ≤ MAX_NODES | Requires good heuristic design; expensive per-node |
| **Beam Search** | None (greedy pruning) | beam_width × depth | Discards paths permanently |

**Key insight from thesis:** No single method dominates across all problems. This is why routing (Chapter 11) matters — the right method depends on problem structure.

---

## 10.13 Summary & Interview Tips

**If asked "Explain A* search":**
1. State the evaluation function: f = g + h
2. Explain the priority queue (min-heap on f)
3. Mention admissibility → optimality guarantee
4. Distinguish tree search vs graph search (visited set)

**If asked "How did you apply A* to LLMs?":**
1. Nodes = partial reasoning traces
2. Edges = LLM continuations at varied temperatures
3. Cost = inverse confidence (uncertain steps are expensive)
4. Heuristic = entity coverage + depth estimate
5. Goal = final answer extracted from trace
6. Key challenge: real branching (K separate API calls)

**If asked "Prove your heuristic is admissible":**
1. State max h at each depth
2. State min path cost (2 × min_step_cost)
3. Show max h < min path cost at every depth
4. For consistency: show max Δh < min step cost

**If asked "What's the computational cost?":**
- Each node expansion = 1 LLM call (or K calls for branching)
- Total calls bounded by MAX_NODES × BRANCH_K
- HotpotQA: ≤ 30 × 3 = 90 calls worst case
- Average: ~12 nodes explored before finding answer
