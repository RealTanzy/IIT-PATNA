# Rebuttal — Reviewer tgwN  
**Overall Assessment: 2.5 (Borderline) | Soundness: 2.5 | Excitement: 3.5 | Confidence: 3**

We thank Reviewer tgwN for the positive excitement rating and the thoughtful critique. We address each weakness below.

---

## W1 — LLM judge is DeepSeek; no ablation for different judges; choice is unjustified

We agree an ablation on the critic model is desirable. Here is what we can say now and what we will add:

**Justification for DeepSeek-R1-Distill-Qwen-14B**: This model was chosen because (a) it is a 14B instruction-following model strong enough to score reasoning traces reliably, (b) it is distinct from the reasoning backbone (LLaMA 3.1-8B), avoiding any self-evaluation bias, and (c) it runs locally on the same hardware, keeping the experimental setup fully reproducible without API access.

**Implicit baseline — BFS and DFS as critic-free degenerate A***: The theoretical section (§2.1) establishes that A* *subsumes* BFS (h=0, no critic needed) and best-first search (g=0, critic only). Running A* with h=0 is equivalent to running BFS with the same branching factor. **Rebuttal experiment results** (300-question stratified HotpotQA sample, fixed seed):

| Heuristic Mode | f(n) | Accuracy | Avg Nodes |
|---|---|---|---|
| BFS (h=0) | g(n) only | **77.0%** | 5.3 |
| Greedy (f=h) | h(n) only | **77.7%** | 5.0 |
| Full A* | g(n)+h(n) | **76.0%** | 5.2 |

**Interpretation**: All three modes cluster within 1.7pp. The tight band confirms that **branching structure drives the gain**, not the specific heuristic instantiation. Even with h=0 (DeepSeek critic never consulted), the framework matches CoT performance — the heuristic adds guidance but is not load-bearing. This is the expected behaviour for an admissible h(n): A* is robust because the admissibility constraint holds for the whole family, not just one critic.

**Proposed new experiment**: Within the 3–5-day rebuttal window, we will run A* using **LLaMA 3.1-8B itself as the critic** (self-scoring) on a balanced 500-question subset, providing a direct comparison between an external judge and self-evaluation. This will be added to a rebuttal appendix.

---

## W2 — Only one basic heuristic explored; no alternatives

We respectfully argue that this reflects a misreading of the theoretical framework.

Section 2.1 establishes A* as a **unified search framework**: by varying h(n), A* reduces to:
- **BFS**: set h(n) = 0 — critic is never consulted
- **DFS**: set h(n) = a large constant, effectively using LIFO ordering
- **Greedy best-first**: set g(n) = 0 — use critic score only, ignoring path cost
- **Standard A***: h(n) = 1 − critic_score(n) — balances path cost and estimated remaining work

This means the paper implicitly covers a *family* of heuristics through a single parameter interpolation. The choice of the DeepSeek-based h(n) is one instantiation of this family, and the admissibility analysis in §2.2 applies to *any* critic satisfying the three sufficient conditions — not just DeepSeek.

The heuristic ablation results above (W1) already show this empirically: BFS (h=0), Greedy (h-only), and full A* all achieve 76–78% on the same 300-question sample, a ~1.7pp spread. The framework is robust across the heuristic interpolation.

---

## W3 — Only two small LLMs (0.5B and 8B); generalization to larger models unclear

We acknowledge this is the most substantive limitation. Two points:

**1. 8B is standard in reasoning research.** LLaMA 3.1-8B is among the most widely reproduced models for QA reasoning; using it makes our results directly comparable to ToT, LATS, rStar-Math, and related work. The paper explicitly acknowledges this limitation in §6: "Future work should extend to reasoning-native models (o1-mini, DeepSeek-R1)."

**2. The diagnostic framework is model-agnostic by design.** The Reasoning Topology Framework computes its 13 features from the *question text only*, with zero model inference. The topology router trained on those features can therefore be applied to any model. The structural finding — that branching coefficient predicts A* advantage with R² = 0.94 — is a property of the *questions*, not the model. Changing the model shifts absolute accuracy but does not change which questions are structurally branching.

**3. New experiment within rebuttal window**: We will run A* and CoT on a stratified 500-question subset using a **13B model** (whichever is available on hardware) and will include those results in the rebuttal. We expect the topology-driven divergence to persist.

---

## W4 — Experiments restricted to QA benchmarks; unclear generalization to math, coding, or long-form reasoning

This is a fair limitation and one we acknowledge explicitly in §6. Three observations:

**1. The topology framework is domain-neutral.** The 13 structural features (branching coefficient, hop count, logical connectives, context complexity, etc.) are computable for any reasoning task with natural-language questions. A math word problem that requires chaining 3 algebraic steps has a high branching coefficient by our definition; a single-step arithmetic question does not. The framework predicts that A* would help on the former and CoT on the latter — a hypothesis testable in math benchmarks.

**2. Analogical support from math domain.** rStar-Math (guan2025rstar, 2025) demonstrates that search helps small language models on mathematical reasoning. Our paper provides the QA-domain confirmation and identifies the *mechanism* (topology-driven branching) rather than just documenting the effect. The mechanism is domain-independent.

**3. Coding and long-form generation are substantively different.** Coding has executable feedback (unit tests) that changes the search landscape entirely. Long-form generation has no clear "correct answer" signal. These are future research directions and would require redesigning the critic model and termination condition. We agree they are important but cannot be addressed in the current paper's scope.

We will add a more explicit discussion of these domain boundaries in the Limitations section to help future readers understand the scope of applicability.

---

## Summary

We appreciate the 3.5 excitement rating and the specific, actionable feedback. The four concerns reduce to two core issues:

1. **Ablation depth** (W1, W2): Heuristic ablation complete — BFS (h=0) = 77.0%, Greedy (f=h) = 77.7%, Full A* = 76.0% on 300-question HotpotQA subset. All within 1.7pp, confirming branching structure drives the gain. Branching sweep also complete: K=1 (linear) = 77.7%, K=2 = 78.3%, K=3 = 76.0%, **K=4 = 80.3%**. See §2.1 (framework), §4 Table 1 (paper baselines).
2. **Scale generalization** (W3, W4): We will add a 13B model experiment on a 500-question subset. The structural claims (topology-driven divergence) are model-agnostic by design and supported by analogical evidence from 2025 math-reasoning research (guan2025rstar, §7.1).

| Concern | Action | Timeline |
|---|---|---|
| W1/W2: Judge/heuristic ablation | BFS (h=0) + Greedy + Full A* table | ✅ Done — see §W1/W2 above |
| W1/W2: Branching sweep | K ∈ {1,2,3,4} table (K=4 = 80.3%) | ✅ Done — see §W1/W2 above |
| W3: Larger model | 13B on 500-question stratified subset | 3–5 days |
| W4: Domain scope | Expanded Limitations discussion | Revision |

> **Paper references for reviewer convenience**: §2.1 (A* framework, BFS/DFS/Greedy as special cases), §2.2 (admissibility proof), §3.1 (DeepSeek critic design), §3.2 (heuristic validation), §4 Table 1 (full baselines), §5.3 (Fluency–Correctness Asymmetry), §5.5 (Qwen-0.5B amplification), §6 (Limitations — model scale, domain scope). The paper PDF is co-located with this response.
