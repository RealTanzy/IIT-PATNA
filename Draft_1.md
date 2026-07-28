# Topology-Aware Adaptive Routing with A* Guided Reasoning for Large Language Models
## Experiment Code Documentation

**Prepared by:** MD Tanzeel Adam Khan

---

This codebase implements A* guided reasoning search for LLM-based problem solving, along with baseline methods (CoT, SC, ToT) and adaptive routing strategies. All experiments were run on code generation (HumanEval, MBPP, CodeContests), mathematical reasoning (GSM8K, MATH), and QA benchmarks (StrategyQA, HotpotQA, LogicQA) across three model scales (0.5B, 8B, 14B).

---

## 1. A* Search Algorithm (Core Contribution)

### Node Representation

```python
@dataclass
class CodeNode:
    plan_steps: list    # Accumulated reasoning steps ["Step 1: ...", "Step 2: ..."]
    code: str           # Generated solution (empty until goal state)
    depth: int          # Current depth in search tree
    g_score: float      # Path cost from root to this node
    h_score: float      # Heuristic estimate of remaining cost
    node_id: int        # Unique identifier for deduplication

    @property
    def f_score(self):
        return self.g_score + self.h_score  # Evaluation function
```

A node is a **goal** when `node.code != ""` (a complete solution has been generated).

### Admissible Heuristic

```python
def compute_code_heuristic(node, problem):
    if node.code:
        return 0.0                        # Goal state: no remaining cost
    return max(0, 3 - node.depth) * 0.3   # Estimated steps to code generation
```

**Admissibility proof:**
- At goal states (code generated), h(n) = 0 (exact).
- For non-goal nodes at depth d, the minimum remaining cost to reach a goal is at most (3 - d) planning steps. Since each step costs at minimum 0.3 in practice, h(n) = max(0, 3-d) x 0.3 never exceeds the true remaining cost.
- h(n) is monotonically non-increasing with depth, satisfying the consistency property.
- Therefore, A* with this heuristic is guaranteed to expand nodes in non-decreasing f-order and find the lowest-cost goal first.

### Search Procedure

```
Initialize: root node (empty plan, h=0.9)
            priority queue (min-heap on f_score)

While heap non-empty AND explored < MAX_NODES (15):
    node = pop minimum f_score from heap
    
    if node.code exists:         # GOAL STATE
        add to goals list
        if 3 goals collected: stop
        continue
    
    if duplicate signature: skip
    
    if depth < 2:               # PLANNING PHASE
        Generate next reasoning step via LLM (T=0.2)
        child.g_score = node.g_score + 1.3
        child.h_score = heuristic(child)
        push child to heap
    
    else:                       # CODE GENERATION PHASE
        Generate complete code from accumulated plan (T=0.2)
        child.g_score = node.g_score + 0.5
        child.h_score = 0.0     # Goal node
        push child to heap

Goal Selection:
    Sort goals by g_score (ascending)
    Return first goal that passes test execution
    Fallback: return lowest g_score goal
```

### Cost Model

| Action | g_score increment | Rationale |
|--------|-------------------|-----------|
| Planning step (depth < 2) | +1.3 | Higher cost reflects uncertainty in reasoning |
| Code generation (depth >= 2) | +0.5 | Lower cost: plan is already formed |

The asymmetric cost encourages the search to progress quickly to code generation once a reasonable plan exists, rather than endlessly refining the plan.

### Deduplication

```python
sig = " | ".join(node.plan_steps) + " || " + node.code[:100]
if sig in visited_sigs: continue
```

Prevents re-expanding equivalent states (same reasoning path leading to same code prefix).

---

## 2. Baseline Methods

### Chain-of-Thought (CoT)

Single-pass generation with no search:
- Temperature: 0.0 (deterministic)
- Prompt: "Solve this problem. Return ONLY the complete function."
- No retry, no evaluation, no branching

### Self-Consistency (SC)

```
Generate 5 independent samples at T=0.7
Test each sample against problem assertions
Correct = (passing_count > 5 // 2)    # Majority vote: >2 must pass
Output = first passing sample (or first generated if none pass)
```

SC exploits the observation that diverse samples at high temperature will converge on the correct answer when the model "knows" the solution, but diverge on hard problems.

### Tree-of-Thought (ToT)

```
1. Generate 3 approaches at T=0.8 (diverse reasoning strategies)
2. Evaluate each approach: LLM scores 1-10 on correctness/completeness
3. Rank by score, select top 2
4. Generate code for each selected approach at T=0.2
5. Test all; return first passing solution
```

ToT separates strategic planning from execution, using explicit evaluation to prune unpromising paths before committing computation to code generation.

---

## 3. Router Decision Logic

Routers decide per-problem whether to use CoT or A*, using different signals.

### Semantic Entropy Router

```
For each problem:
    Generate 5 code samples at T=0.7
    Count unique solutions (normalized)
    if unique_count >= 3: route to A*    # High disagreement = hard
    else: route to CoT                    # Low disagreement = easy
```

Intuition: When the model produces many distinct solutions, it lacks confidence — A*'s structured search helps. When solutions converge, the model is confident — CoT suffices.

### LLM-as-Critic Router

```
Prompt (to 14B judge model):
    "Compare Solution A (CoT output) vs Solution B (A* output).
     Evaluate: correctness entailment, logical errors, edge cases.
     Return: {"better_trace": "A" or "B"}"

if better_trace == "B": route to A*
else: route to CoT
```

### LLM-as-Judge Router

```
Prompt (to 14B judge model):
    "Compare two candidate solutions. Evaluate on:
     faithfulness, logical validity, completeness.
     Return: <winner>A</winner> or <winner>B</winner>"

if winner == "B": route to A*
else: route to CoT
```

Difference from Critic: Judge evaluates holistic quality with emphasis on faithfulness to problem requirements; Critic focuses on flaw detection.

### Random Forest Router

```
Feature extraction (8 features from problem text):
    1. Word count
    2. Logical connectives (if, then, because, since, but, unless...)
    3. Negation words (not, no, never, without, cannot...)
    4. Comparison words (more, less, greater, than...)
    5. Condition count (if/else/elif occurrences)
    6. Example count (>>>, assert, Example)
    7. Code keywords (return, list, array, string, sort...)
    8. Special characters ([]{}()<>=+-*/)

Label: y = 1 if (A* correct AND CoT wrong), else 0
Model: RandomForest(n_estimators=300, max_depth=5)
Evaluation: 5-fold Stratified Cross-Validation
Decision: predict(features) == 1 → route to A*, else CoT
```

No LLM calls — purely structural features of the problem text predict when A* outperforms CoT.

---

## 4. Experimental Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| Test timeout | `signal.alarm(10)` — kills code execution after 10 seconds (prevents infinite loops) |
| Incremental save | Results appended to JSON after every problem (survives crashes/disconnects) |
| Resume logic | Loads completed task_ids on startup; skips already-done problems |
| API retry | 3 attempts with exponential backoff on timeout/rate-limit |
| Deduplication | A* skips nodes with identical (plan, code_prefix) signatures |

---

## 5. Key Parameters

| Parameter | Value | Method | Rationale |
|-----------|-------|--------|-----------|
| MAX_NODES | 15 | A* | Exploration budget (compute bound) |
| MAX_DEPTH | 5 | A* | Prevents infinite deepening |
| BRANCH_K | 1 | A* | Single branch per expansion (no-hop variant) |
| SC_SAMPLES | 5 | SC | Standard in literature (Wang et al., 2023) |
| SC temperature | 0.7 | SC | High enough for diversity, low enough for coherence |
| TOT_BRANCHES | 3 | ToT | Manageable evaluation cost |
| TOT temperature | 0.8 (plan), 0.2 (code) | ToT | Diverse exploration, deterministic execution |
| SE threshold | >= 3 unique | Router | Majority-disagreement signals difficulty |
| RF trees | 300, depth 5 | Router | Sufficient capacity without overfitting |
| RF CV folds | 5 | Router | Standard stratified evaluation |

---

## 6. File Reference

| Script | Purpose |
|--------|---------|
| `run_14b_humaneval_mbpp_cc.py` | 14B CoT + A* on code benchmarks |
| `run_8b_astar_sc_tot_cc.py` | 8B A* + SC + ToT on CodeContests |
| `run_8b_cot_cc_only.py` | 8B CoT on CodeContests (fixed rerun) |
| `run_14b_sc_tot.py` | 14B SC + ToT on code benchmarks |
| `run_05b_cot_astar.py` | 0.5B CoT + A* on code benchmarks |
| `run_05b_sc.py` | 0.5B SC on code benchmarks |
| `run_05b_tot_and_cc.py` | 0.5B ToT + CodeContests all methods |
| `run_gsm8k_math_14b.py` | 14B CoT + A* on GSM8K and MATH |
| `run_gsm8k_math_05b.py` | 0.5B CoT + A* on GSM8K and MATH |
| `run_sc_tot_math_14b_fast.py` | 14B SC + ToT on math (concurrent) |
| `run_sc_tot_math_05b.py` | 0.5B SC + ToT on math |
| `run_routers_code.py` | All 4 routers for 8B code benchmarks |
| `run_routers_14b_code.py` | All 4 routers for 14B code |
| `run_routers_05b_code.py` | All 4 routers for 0.5B code |
| `run_routers_math.py` | GSM8K/MATH routers (parameterized by scale) |
| `run_qa_judge.py` | LLM-as-Judge for QA datasets |
| `extract_problem_prompts.py` | Extract raw prompts from HuggingFace datasets |
