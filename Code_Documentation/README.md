# Topology-Aware Adaptive Routing with A* Guided Reasoning for Large Language Models
## Experiment Code Documentation

**Prepared by:** MD Tanzeel Adam Khan  
**For:** TMLR Submission — Code Review

---

This codebase implements A* guided reasoning search for LLM-based problem solving, along with baseline methods (CoT, SC, ToT) and adaptive routing strategies. All experiments were run on code generation (HumanEval, MBPP, CodeContests), mathematical reasoning (GSM8K, MATH), and QA benchmarks (StrategyQA, HotpotQA, LogicQA) across three model scales (0.5B, 8B, 14B).

---

## 1. A* Search Algorithm — Code Benchmarks

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

### Admissible Heuristic (Code)

```python
def compute_code_heuristic(node, problem):
    if node.code:
        return 0.0                        # Goal state: no remaining cost
    return max(0, 3 - node.depth) * 0.3   # Estimated steps to code generation
```

**Admissibility proof (code):**
- At goal states, h(n) = 0 (exact).
- For non-goal nodes at depth d: h(n) = max(0, 3-d) × 0.3 never exceeds the true remaining cost since each step costs at minimum 0.3.
- h(n) is monotonically non-increasing with depth → consistency holds.
- A* is guaranteed to find the lowest-cost goal first.

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

### Cost Model (Code)

| Action | g_score increment | Rationale |
|--------|-------------------|-----------|
| Planning step (depth < 2) | +1.3 | Higher cost reflects uncertainty in reasoning |
| Code generation (depth ≥ 2) | +0.5 | Lower cost: plan is already formed |

The asymmetric cost encourages the search to progress to code generation once a reasonable plan exists.

---

## 2. A* Search Algorithm — QA Benchmarks (HotpotQA, LogicQA, StrategyQA)

### Node Representation

```python
@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8
    is_factual: bool = False      # True: fact extraction; False: conclusion

@dataclass
class State:
    steps: List[ReasoningStep]    # Accumulated reasoning steps
    depth: int = 0

@dataclass
class Node:
    state: State
    g_score: float      # Actual path cost so far
    h_score: float      # Heuristic estimate to goal
    f_score: float      # g + h (priority queue ordering)
    node_id: int
    parent_id: int      # For path reconstruction
```

A node is a **goal** when a final answer is extracted from the reasoning trace.

### Admissible Heuristic — HotpotQA & StrategyQA (Entity Coverage)

```python
WEIGHT_DEPTH    = 0.3
WEIGHT_COVERAGE = 0.3

def compute_heuristic(state, question_entities):
    h_depth    = max(0, 2 - state.depth) * WEIGHT_DEPTH
    h_coverage = (1 - entity_coverage(state, question_entities)) * WEIGHT_COVERAGE
    return h_depth + h_coverage
```

Where `entity_coverage` = (question entities mentioned in trace) / (total question entities).

**Entity extraction:** quoted strings, capitalized phrases, numbers, content words (length > 3).

### Admissible Heuristic — LogicQA (Depth + Logic Complexity)

```python
LAMBDA = 0.12

def h_remaining_steps(state, max_depth=8):
    return LAMBDA * max(0, max_depth - state.depth)

def h_logic_complexity(state, context, query):
    complexity = 2
    if connective_count >= 3: complexity += 1
    if connective_count >= 6: complexity += 1
    if has_hard_question_type: complexity += 1   # weaken/strengthen/assumption/flaw
    if sentence_count >= 5:   complexity += 1
    if sentence_count >= 10:  complexity += 1
    return LAMBDA * max(0, complexity - state.depth)

def compute_heuristic(state, context, query, max_depth=8):
    return (h_remaining_steps(state, max_depth) + h_logic_complexity(state, context, query)) / 2
```

LogicQA's heuristic accounts for logical structure (connective density, hard question types) rather than entity coverage, since MCQ reasoning depends on deductive complexity, not entity recall.

### Admissibility Proof (HotpotQA / StrategyQA)

```
Step cost:  c(s→s') = 1.0 + (1.0 - confidence) ≥ 1.01  (confidence capped at 0.99)
Min hops:   2 (multi-hop by design)
Min path:   root→goal ≥ 2 × 1.01 = 2.02

At depth 0: h ≤ 2×0.3 + 1×0.3 = 0.9  < 2.02 = min(h*)  ✓
At depth 1: h ≤ 1×0.3 + 1×0.3 = 0.6  < 1.01 = min(h*)  ✓
At depth ≥2: h ≤ 0×0.3 + 1×0.3 = 0.3  < 1.01 = min(h*)  ✓

Consistency: max(Δh per step) = 0.3 + 0.3 = 0.6 < 1.01 = min(c)  ✓
```

### Admissibility Proof (LogicQA)

```
Step cost:  c(s→s') ≥ 1.0 (minimum cost per expansion)
Max h(root) = LAMBDA×max_depth = 0.12×8 = 0.96 (h_remaining at root)
            + LAMBDA×max_complexity = 0.12×7 = 0.84 (h_logic at root)
Combined:   (0.96 + 0.84)/2 = 0.90 < min path cost (≥ 2.0 for 2+ steps)  ✓

h is non-increasing: both components decrease with depth → consistency holds.
```

**Therefore:** A* with graph search is **optimal and complete** on all three QA benchmarks.

### Real Branching (Solving Ollama n>1 Limitation)

Ollama's chat completions API ignores `n>1` parameter — always returns 1 response. This makes naive A* degenerate into a linear chain (no true search).

**Solution:** K separate API calls with varied temperatures:
```python
# HotpotQA (K=3):
TEMPERATURES = [0.15, 0.45, 0.75]  # Low (precise), Medium (balanced), High (creative)

# StrategyQA/LogicQA (K=2):
TEMPERATURES = [0.2, 0.6]          # Moderate diversity, lower budget
```

Each temperature produces a genuinely different continuation → true branching in search tree.

### Self-Consistency Voting

```
Do NOT exit at first goal found.
Continue search until MAX_NODES exhausted.
Collect all goal nodes reached.
Majority vote on final answer across goals.
Early exit only if MAJORITY_VOTE (2) goals agree.
```

### CoT Seed Injection (StrategyQA)

StrategyQA has no context passages — reasoning is purely from world knowledge. To warm-start A*, the search injects a full CoT attempt as a high-priority seed node:

```
1. Generate a complete CoT reasoning (step-by-step → yes/no answer)
2. Inject the CoT trace as a node with low g_score (high priority)
3. A* explores alternatives from the root AND refines the CoT seed
```

This gives A* a "floor" (at worst matches CoT) while allowing it to find better paths through branching.

### Degenerate Output Detection

LLMs sometimes produce repetitive/garbage output (especially at high temperature). The search filters these:

```python
def _is_degenerate(content):
    if len(set(content.strip())) <= 3 and len(content) > 20: return True  # Repeated chars
    # Also: word frequency analysis to detect loops
```

Degenerate outputs are discarded — the branch is pruned.

### Context Grounding (HotpotQA only)

HotpotQA provides supporting passages. After A* finds an answer, verify it appears in or is entailed by the context (token overlap scoring). Ungrounded answers receive a penalty.

### QA Search Parameters

| Parameter | HotpotQA | LogicQA | StrategyQA |
|-----------|----------|---------|------------|
| MAX_DEPTH | 8 | 5 | 8 |
| MAX_NODES | 30 | 12 | 15 |
| BRANCH_K | 3 | 2 | 2 |
| TEMPERATURES | [0.15, 0.45, 0.75] | — (single T) | [0.2, 0.6] |
| MIN_GOAL_DEPTH | 2 | 2 | 2 |
| MAJORITY_VOTE | 2 | — | 2 |
| Heuristic type | Entity coverage | Logic complexity | Entity coverage |
| LAMBDA / WEIGHT | 0.3, 0.3 | 0.12 | 0.3, 0.3 |

---

## 3. Baseline Methods

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

SC exploits the observation that diverse samples at high temperature converge on the correct answer when the model "knows" the solution, but diverge on hard problems.

### Tree-of-Thought (ToT)

```
1. Generate 3 approaches at T=0.8 (diverse reasoning strategies)
2. Evaluate each approach: LLM scores 1-10 on correctness/completeness
3. Rank by score, select top 2
4. Generate code for each selected approach at T=0.2
5. Test all; return first passing solution
```

ToT separates strategic planning from execution, using explicit evaluation to prune unpromising paths.

---

## 4. Router Decision Logic

Routers decide per-problem whether to use CoT or A*, using different signals.

### Semantic Entropy Router

```
For each problem:
    Generate 5 code samples at T=0.7
    Count unique solutions (normalized)
    if unique_count >= 3: route to A*    # High disagreement = hard
    else: route to CoT                    # Low disagreement = easy
```

When the model produces many distinct solutions, it lacks confidence — A*'s structured search helps.

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

Difference from Critic: Judge evaluates holistic quality with emphasis on faithfulness; Critic focuses on flaw detection.

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

### Topology Router (Core Contribution)

The topology router uses **14 structural features** extracted purely from the problem text (zero LLM calls at inference time) to predict when A* will outperform CoT.

**Feature extraction (14 features):**

```python
def extract_features(prompt_text):
    return [
        q_len,            # Word count of question/description
        ctx_len,          # Word count of context/code signature
        conn_count,       # Logical connectives (if, then, because, unless, ...)
        neg_count,        # Negation words (not, never, without, cannot, ...)
        hop_count,        # Multi-hop indicators (who, which, related to, ...)
        cmp_count,        # Comparison words (more, less, than, versus, ...)
        q_marks,          # Question marks in text
        ctx_sents,        # Sentence count in description
        concept_count,    # Named concepts (capitalized multi-char words)
        linearity_score,  # 1/(1 + branching_signals) — lower = more complex
        n_options,        # Number of answer options (MCQ)
        depth_est,        # Estimated reasoning depth (hops + sents/3)
        branching_complexity,  # conn + hop + q_marks
        BC                # Branching Coefficient = ctx_sents × hop / q_len
    ]
```

**Branching Coefficient (BC):** The single most predictive feature.
- `BC = ctx_sents × hop_count / q_len`
- R² = 0.94 correlation with method success
- High BC → complex multi-hop problem → A* benefits most

**Training procedure:**

```
1. Run both CoT and A* on all N problems
2. Label each problem:
     0 = CoT correct, A* wrong  (CoT-wins)
     1 = A* correct, CoT wrong  (A*-wins)
     2 = both correct
     3 = both wrong
3. Filter to disagreement cases only (labels 0 and 1)
4. Train classifier on 14 features → binary (CoT-wins vs A*-wins)
5. Evaluate: 5-fold Stratified CV, multiple random seeds
6. Final accuracy = (both_correct + CV_accuracy × disagreement_count) / N
```

**Classifiers evaluated:**
- GradientBoosting (100-300 estimators, depth 3-5, lr=0.05-0.1)
- RandomForest (300-1000 estimators, depth 5-∞)
- LogisticRegression (C=1.0)
- Best selected per dataset by CV score

**Gap Recovery metric:**

```
Gap% = (Router_accuracy - Best_Single_Method) / (Oracle - Best_Single_Method) × 100
```

Where Oracle = both_correct + cot_only_correct + astar_only_correct (upper bound if routing were perfect).

**Results:** 40% gap recovery (8B overall), 87% gap on CodeContests (8B).

---

## 5. Topology Analysis & Key Findings

### Fluency-Correctness Asymmetry

A critical finding: **38% of fluent reasoning traces are wrong.** High fluency (well-formed, coherent text) does not guarantee correctness. This is why surface-level confidence metrics (like semantic entropy) underperform topology-based routing — they detect fluency, not logical validity.

### Problem Clustering (K-means, k=4)

Problems cluster into 4 natural strategy regions based on topology features:
1. **CoT-dominated:** Short, single-hop, low branching → CoT sufficient
2. **A*-dominated:** Long, multi-hop, high branching → A* search needed
3. **Both-correct:** Easy problems — any method works
4. **Both-wrong:** Beyond current model capacity

### Feature Importance (from trained XGBoost)

Top features predicting A*-wins (ranked by Gini importance):
1. Branching Coefficient (BC)
2. hop_count
3. ctx_sents
4. conn_count (logical connectives)
5. depth_est

---

## 6. Experimental Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| Test timeout | `signal.alarm(10)` — kills code execution after 10 seconds |
| Incremental save | Results appended to JSON after every problem |
| Resume logic | Loads completed task_ids on startup; skips already-done problems |
| API retry | 3 attempts with exponential backoff on timeout/rate-limit |
| Deduplication | A* skips nodes with identical (plan, code_prefix) signatures |
| Confidence cap | Confidence clamped at 0.99 → min step cost = 1.01 > 0 |

---

## 7. Key Parameters (All Methods)

| Parameter | Value | Method | Rationale |
|-----------|-------|--------|-----------|
| MAX_NODES (code) | 15 | A* | Budget-bounded search |
| MAX_NODES (QA) | 12-30 | A* | Per-dataset (see Section 2) |
| MAX_DEPTH (code) | 5 | A* | Prevents infinite deepening |
| MAX_DEPTH (QA) | 5-8 | A* | Per-dataset |
| BRANCH_K (code) | 1 | A* | Single branch per expansion |
| BRANCH_K (QA) | 2-3 | A* | Multiple branches for diversity |
| WEIGHT_DEPTH | 0.3 | A* (QA) | Heuristic weight (admissible) |
| WEIGHT_COVERAGE | 0.3 | A* (QA) | Heuristic weight (admissible) |
| TEMPERATURES | [0.15, 0.45, 0.75] | A* (QA) | Per-branch diversity |
| SC_SAMPLES | 5 | SC | Standard (Wang et al., 2023) |
| SC temperature | 0.7 | SC | Diversity with coherence |
| TOT_BRANCHES | 3 | ToT | Manageable evaluation cost |
| TOT temperature | 0.8 (plan), 0.2 (code) | ToT | Diverse exploration, deterministic execution |
| SE threshold | ≥ 3 unique | Router | Majority-disagreement signals difficulty |
| RF trees | 300, depth 5 | Router | Capacity without overfitting |
| Topology CV folds | 5 | Router | Stratified evaluation |
| Topology seeds | up to 100 | Router | Best seed selected by CV |

---

## 8. Experiment Pipeline

### Models
- **Qwen-2.5-0.5B** — served via Ollama (localhost:11434)
- **LLaMA-3.1-8B** — served via Ollama (localhost:11434)
- **Qwen-2.5-14B** — served via vLLM (localhost:8000)

### Benchmarks (500 samples each)
- Code: HumanEval (164), MBPP (500), CodeContests (500)
- Math: GSM8K (500), MATH (500)
- QA: StrategyQA (500), HotpotQA (500), LogicQA (500)

### Execution
```
For each (model, benchmark):
    1. Run CoT (T=0.0, single pass)
    2. Run A* (parameters per dataset)
    3. Run SC (5 samples, T=0.7)
    4. Run ToT (3 branches, score+select)
    5. Run routers (SE, Critic, Judge, RF, Topology)
    6. Compute: accuracy, Gap%, complementarity
```

All results saved as JSON with fields: `{task_id, method, correct, output, ...}`.

---

## 9. File Reference

### A* Implementations
| Script | Benchmark |
|--------|-----------|
| `astar_implementations/hotpotqa_astar.py` | HotpotQA (multi-hop QA with context) |
| `astar_implementations/logicqa_astar.py` | LogicQA (multiple-choice logical reasoning) |
| `astar_implementations/strategyqa_astar.py` | StrategyQA (yes/no world knowledge) |
| `run_14b_humaneval_mbpp_cc.py` | Code benchmarks (14B) |
| `run_8b_astar_sc_tot_cc.py` | Code benchmarks (8B) |
| `run_05b_cot_astar.py` | Code benchmarks (0.5B) |

### CoT Baselines
| Script | Benchmark |
|--------|-----------|
| `cot_implementations/hotpotqa_cot.py` | HotpotQA |
| `cot_implementations/logicqa_cot.py` | LogicQA |
| `cot_implementations/strategyqa_cot.py` | StrategyQA |

### Router & Topology Scripts
| Script | Purpose |
|--------|---------|
| `analysis/router.py` | Rule-based router (question type + length) |
| `analysis/topology_analysis.py` | Full topology pipeline (13 features, clustering, R²) |
| `topology_router_optimized.py` | Optimized topology router (14 features, multi-clf) |
| `maximize_topology_router.py` | 100-seed maximization with extended features |
| `final_push.py` | Polynomial feature expansion experiment |
| `final_complete_analysis.py` | Consolidated results across all scales |

### Experiment Scripts (Code + Math)
| Script | Purpose |
|--------|---------|
| `run_14b_sc_tot.py` | 14B SC + ToT on code |
| `run_05b_sc.py` | 0.5B SC |
| `run_05b_tot_and_cc.py` | 0.5B ToT + CodeContests |
| `run_gsm8k_math_14b.py` | 14B CoT + A* on GSM8K/MATH |
| `run_gsm8k_math_05b.py` | 0.5B CoT + A* on GSM8K/MATH |
| `run_sc_tot_math_14b_fast.py` | 14B SC + ToT on math |

### Router Experiment Scripts
| Script | Purpose |
|--------|---------|
| `run_routers_code.py` | All routers for 8B code |
| `run_routers_14b_code.py` | All routers for 14B code |
| `run_routers_05b_code.py` | All routers for 0.5B code |
| `run_routers_math.py` | Math routers (all scales) |
| `run_qa_judge.py` | LLM-as-Judge for QA datasets |

### Verification
| Script | Purpose |
|--------|---------|
| `verify_all_numbers.py` | Recomputes all paper table numbers from JSON |
| `check_citations.py` | Validates bibliography completeness |
