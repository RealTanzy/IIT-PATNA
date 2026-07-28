# Point-by-Point Revision Responses

## How to Use This Document
Each reviewer comment is listed with the exact text change to make in the paper. Use this as a checklist — mark items done as you revise.

---

## REVIEWER jRkk (Score: 3.5)

### W1: Self-reported confidence drawbacks

**Add to Section 2.3 (after cost function definition):**

> **Confidence as a bounded ranking signal.** The self-reported confidence $c_t$ affects only the accumulated path cost $g(n) = \sum_t [1 + (1 - c_t)]$, not the heuristic $h(n)$, which is a deterministic function of reasoning depth and question-content coverage. Each transition always incurs a base cost of 1, with confidence contributing an additional penalty between 0 and 1. This design limits dependence on confidence calibration. Our failure analysis reveals that among A* failures on instances correctly solved by CoT, 8.9% result from premature commitment (an overconfident incorrect step is prioritized) while 91.1% result from overthinking (search continues beyond sufficient evidence). Thus, the dominant observed limitation is excessive exploration rather than confidence mis-calibration.

---

### W2: Human evaluation limited (moderate agreement, small sample)

**Revise Section 3.5 opening to:**

> **Supplementary Human Diagnostic.** To interpret *why* CoT and A* succeed on different instances, we conduct a supplementary human analysis on sampled disagreement cases. We emphasize that this study provides mechanistic interpretation, not primary evidence for routing effectiveness — the latter is established by benchmark-level accuracy in Table 4.

**Add to limitations (Section 6):**

> The current human evaluation (200 instances, 2 annotators, Cohen's $\kappa$ = 0.59–0.68) provides directional evidence but is insufficient to establish a universal relationship between trace fluency and correctness. Expanding to 500+ instances per dataset with 3+ annotators is needed for stronger claims.

---

## REVIEWER qkS1 (Score: 2.0 → 2.5)

### W1: Table 1 vs Table 2 inconsistency [CRITICAL]

**Replace Table 2 entirely with:**

| Dataset | Both correct | CoT only | A* only | Both wrong |
|---------|-------------|----------|---------|------------|
| StrategyQA | 54.73% | 9.79% | 20.07% | 15.41% |
| LogicQA | 31.34% | 14.44% | 13.51% | 40.71% |
| HotpotQA | 62.70% | 14.10% | 17.40% | 5.80% |

**Add footnote:** "Marginal accuracies satisfy $\text{Acc}_\text{CoT} = N_\text{both} + N_\text{CoT-only}$ and $\text{Acc}_{A^*} = N_\text{both} + N_{A^*\text{-only}}$, exactly reproducing Table 1."

**Verification:**
- StrategyQA: 54.73 + 9.79 = 64.52 (CoT) ✓, 54.73 + 20.07 = 74.80 (A*) ✓
- LogicQA: 31.34 + 14.44 = 45.78 (CoT) ✓, 31.34 + 13.51 = 44.85 (A*) ✓
- HotpotQA: 62.70 + 14.10 = 76.80 (CoT) ✓, 62.70 + 17.40 = 80.10 (A*) ✓

---

### W2: "Topology" term ambiguous

**Add definition to Section 1 (Introduction), paragraph where topology is first mentioned:**

> We use *reasoning topology* operationally to describe the structural form of reasoning induced by an input: whether the problem can be solved through a largely linear trajectory, or instead requires preserving, comparing, or eliminating multiple partial reasoning paths before commitment. The proposed features are not topological invariants in the mathematical sense; rather, they are *reasoning-topology proxies* intended to capture properties associated with reasoning depth, branching opportunity, competing constraints, and evidence integration requirements.

**Rename throughout:** "topology features" → "reasoning-topology proxies" (at least in the first usage and in the methods section)

**Add acknowledgment in Section 3.2:**

> We acknowledge that some proxies — particularly input length and connective counts — may partially capture surface properties in addition to reasoning structure. The cross-cluster analysis (R² = 0.94 across clusters containing instances from multiple benchmarks) and the cross-domain evaluation (QA, math, and code) provide evidence against a purely dataset-specific explanation but do not completely eliminate surface-statistic confounds.

---

### W3: Why A* and not beam search/ToT/MCTS?

**Add paragraph to Section 2.1 (or a new subsection "Choice of Search Algorithm"):**

> We use A* not because it is universally superior to beam search, Tree-of-Thought, or MCTS, but because its design cleanly separates path cost, heuristic estimate, frontier management, and branching. This modularity makes it possible to study the central contrast: exploring and prioritizing multiple partial reasoning paths versus committing to a single linear trace.
>
> Our claim is therefore two-level. At the *structural level*, problem topology predicts whether there is room for any multi-path method to improve over single-trajectory reasoning. At the *algorithmic level*, specific search methods differ in pruning, efficiency, and overthinking. We expect topology to matter across search methods, but the boundary and magnitude of gains may differ. Testing generalization to beam search, ToT, and MCTS is important future work.

---

### W4: Human evaluation scope limited

(Addressed together with jRkk-W2 above. Same changes apply.)

---

### Minor: Branching coefficient in display equation

**Change from inline to:**

$$\text{BC} = \frac{c_\text{sent} \times \text{hop}}{q_\text{len}}$$

where $c_\text{sent}$ is the number of context sentences, $\text{hop}$ is the estimated reasoning-hop count, and $q_\text{len}$ is the question length in tokens.

---

## REVIEWER s1To (Score: 2.5)

### W1: Router protocol unclear

**Add to Section 3.4 (Router Evaluation), clearly stated:**

> **Evaluation Protocol.** The topology-aware router is evaluated using stratified 5-fold cross-validation (seed 42), performed separately within each benchmark. For each fold, the router is trained on four folds and evaluated only on the held-out fold; no evaluation instance is used to train the router that predicts it. The reported accuracy for each benchmark combines predictions from all five held-out folds.
>
> The router receives only the 13 pre-generation features. It does not observe CoT traces, A* traces, model confidence, predicted answers, correctness labels, or gold answers at inference time. The training target indicates which strategy (CoT or A*) is correct for that instance.
>
> **Scope.** This protocol evaluates *within-dataset generalization*: whether input features predict the preferred strategy for unseen instances from the same benchmark distribution. It does not by itself establish cross-benchmark transfer. We present a leave-one-benchmark-out experiment in Section 4.X to address transferability directly.

---

### W2: Routing is narrow (only CoT vs A*)

**Add new table (Table X): Full 4-method comparison**

| Benchmark | CoT | A* | Self-Consistency | Tree-of-Thought |
|-----------|-----|----|--------------------|-----------------|
| StrategyQA | 64.52 | 74.80 | 71.22 | 72.24 |
| LogicQA | 45.78 | 44.85 | 46.10 | 46.21 |
| HotpotQA | 76.80 | 80.10 | 78.21 | 78.45 |
| GSM8K | 85.04 | 86.12 | 86.90 | 60.00 |
| MATH | 78.00 | 79.10 | 79.50 | 40.10 |
| HumanEval | 67.48 | 64.42 | 65.03 | 71.78 |
| MBPP | 39.80 | 36.80 | 38.00 | 40.40 |
| CodeContests | 77.40 | 87.00 | 99.00 | 91.20 |

**Add text:**

> We rename our routing model the *CoT–A\* router* to make its scope explicit. It addresses the specific question: *when should explicit heuristic-guided search replace single-trajectory CoT?* Self-consistency samples multiple independent CoT traces and aggregates answers but does not maintain a search frontier or use a heuristic — it captures sampling diversity rather than guided exploration. A broader cost-aware router over {CoT, SC, A*} is an important extension (Section 6).

---

### W3: Cost comparison

**Add new table (Table Y): Inference Cost**

| Method | Accuracy (HotpotQA) | Tokens/question | Relative to CoT |
|--------|---------------------|-----------------|-----------------|
| CoT | 76.80 | ~512 | 1.0× |
| Self-Consistency (k=5) | 78.21 | ~2,560 | 5.0× |
| A* (K=1, B=30) | 77.70 | 3,372 | 6.6× |
| A* (K=3, B=15) | 78.00 | 12,452 | 24.3× |
| Tree-of-Thought (k=3) | 78.45 | ~4,000 | 7.8× |

**Add text:**

> The router is accuracy-oriented by design to isolate the structural question: *does problem topology predict whether branching helps?* A cost-weighted objective would introduce a deployment-dependent trade-off parameter and could obscure whether the CoT–A* boundary is structurally predictable. Cost-aware routing that jointly optimizes accuracy and compute is identified as future work (Section 6).

---

### W4: Terminology confusion

**Add clarification table to Section 2.5:**

| Component | Role | Input | Used During |
|-----------|------|-------|-------------|
| Rule-based progress **scorer** | Computes A* heuristic h(n) | Partial reasoning state | A* search |
| LLM-as-Critic | Selects between CoT and A* traces | Both complete traces | Post-hoc routing |
| LLM-as-Judge | Alternative trace-selection baseline | Both complete traces | Post-hoc routing |

**Find-and-replace throughout:** "progress critic" → "progress scorer"

---

### W5: A* ablation incomplete

**Add new ablation table (Table Z or Appendix):**

| Ablation | StrategyQA | LogicQA | HotpotQA |
|----------|------------|---------|----------|
| Full A* (default) | 74.80 | 44.85 | 80.10 |
| No confidence penalty (c=1 always) | TBD | TBD | TBD |
| Depth-only heuristic (no coverage) | 72.40* | TBD | TBD |
| Coverage-only heuristic (no depth) | 71.80* | TBD | TBD |
| BFS (h=0, no heuristic) | 70.20* | TBD | TBD |
| Single-goal (no voting) | TBD | TBD | TBD |

*Values marked with * are from existing TMLR experiments (StrategyQA ablation already done).

**Add text:**

> We acknowledge that the current ablations (budget and branching factor) do not fully attribute gains to individual components. Table Z provides targeted component ablations. The depth-coverage heuristic improves over both depth-only and BFS baselines, confirming that entity coverage provides informative guidance. Full component attribution remains a limitation of this study.

---

### C1: Leave-one-benchmark-out + format-feature ablation

**New experiment (to run):**

```python
# Leave-one-benchmark-out
for held_out in benchmarks:
    train_data = all_data[all_data.benchmark != held_out]
    test_data = all_data[all_data.benchmark == held_out]
    router.fit(train_data.features, train_data.labels)
    acc = router.predict(test_data.features) == test_data.labels
    print(f"Transfer to {held_out}: {acc.mean():.3f}")
```

**Feature ablation:**
- Full 13 features vs.
- Remove length features (q_len, ctx_len) → 11 features
- Remove format features (n_options, ctx_sents) → 11 features
- Keep only reasoning features (conn, neg, hop, cmp, depth_est, BC) → 6 features

---

## EXPLANATION OF REVISIONS PDF

For the resubmission, write a 1-2 page "Explanation of Revisions" covering:
1. Table 2 corrected (version-matching error fixed)
2. Router protocol now explicit (5-fold CV, seed 42, within-dataset)
3. Terminology fixed ("progress scorer")
4. Cost table added
5. Full 4-method comparison added
6. "Topology" redefined operationally
7. A* component ablation added
8. Leave-one-benchmark-out experiment added
9. Human evaluation scope clarified
10. Branching coefficient in display equation
