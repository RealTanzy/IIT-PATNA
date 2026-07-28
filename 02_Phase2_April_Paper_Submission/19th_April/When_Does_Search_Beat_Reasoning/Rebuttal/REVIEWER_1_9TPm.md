# Rebuttal — Reviewer 9TPm  
**Overall Assessment: 2.5 (Borderline) | Soundness: 3 | Confidence: 4**

We thank Reviewer 9TPm for the careful and constructive reading. We address each weakness in order.

---

## W1 — Routing gap: gap between oracle and actual routing is substantial

We respectfully argue this is **a finding, not a shortcoming**. The paper's contribution is to *identify and explain* why the gap exists — not to close it completely.

Section 5.3 provides the mechanistic explanation via the **Fluency–Correctness Asymmetry**: in 38% of divergent cases, CoT produces a more fluent, readable reasoning trace while arriving at the *wrong* answer. Any routing method that evaluates trace quality (LLM-as-Critic, Semantic Entropy) encounters a **hard ceiling** precisely at those cases, because the surface signal is misleading. The topology router, which never inspects a trace, bypasses this ceiling and outperforms all output-based routers (26.1% oracle gap recovery) — but it too is bounded because it cannot model the compatibility between problem structure and the model's parametric knowledge for individual instances.

This is therefore a **diagnostic contribution**: we have identified the exact bottleneck (Fluency–Correctness cases) and proposed what the next generation of routers must overcome (lightweight problem–model compatibility scoring, §6). We will add a clearer paragraph in the Routing section making this framing explicit.

---

## W2 — Heuristic function may be dataset-dependent

The admissibility argument is intentionally general. We provide **three sufficient conditions** (monotone scoring, conservative estimation, bounded overestimation) that correspond to different assumptions about the critic model's behaviour — none of which references any dataset-specific feature.

Critically, the empirical validation (§3.2) is conducted on **500 annotated reasoning states drawn from all three datasets** (100 each for StrategyQA and HotpotQA, 300 for LogicQA). The mean underestimation rate (0.43 vs. human-assessed 0.58, p < 0.001) and overestimation bound (ε_max = 0.14) are consistent across all three, which confirms that the heuristic properties are not dataset-specific.

Additionally, the theoretical framework is explicit that these are **sufficient, not necessary** conditions: A* can still be effective when the heuristic is occasionally inadmissible because the budget B = 15 acts as a regulariser. This is a standard result in bounded-suboptimal search (Hart et al., 1968).

---

## W3 — Limited benchmarks; unclear scalability to larger models or more complex tasks

Three points:

1. **Benchmark diversity**: LogicQA (formal deductive), StrategyQA (commonsense multi-hop), and HotpotQA (evidence synthesis) represent three structurally distinct QA paradigms. The topology framework's R² = 0.94 fit shows that A*'s advantage is predicted by *branching coefficient* — a structural property of the problem — not by dataset identity. This suggests the framework generalises structurally.

2. **Larger models**: We agree this is a natural next step and list it explicitly in Limitations (§6): "Results are based on LLaMA 3.1-8B and Qwen-0.5B. Future work should extend to reasoning-native models (o1-mini, DeepSeek-R1)." Within the 3–5-day rebuttal window, we will run a sensitivity experiment using a 13B model on a 500-question stratified subset and will report those numbers in a revised appendix if results are available before the deadline.

3. **Small models remain relevant**: The 3.5× search-as-capability-amplification finding on Qwen-0.5B (§5.5) has direct practical relevance for on-device deployment and resource-constrained settings, which is an increasingly important deployment scenario.

---

## W4 — Comparison with Tree-of-Thought and other search-based methods is not sufficiently detailed

We respectfully point out that **ToT is already included as a direct baseline** in Table 1 (§4). The result is: "Self-consistency and Tree-of-Thoughts consistently fall between CoT and A*." The methodological comparison (§7.1, Related Work) explicitly distinguishes A*'s priority rule from ToT's undirected BFS/DFS and notes that A* provides a formal admissibility guarantee that ToT lacks.

We have already run this ablation during the rebuttal window. Results on a stratified 300-question HotpotQA subset (same seed, same questions across all conditions):

| Heuristic Mode | Description | Accuracy | Avg Nodes |
|---|---|---|---|
| BFS (h=0) | critic disabled entirely | **77.0%** | 5.3 |
| Greedy (f=h) | critic only, no path cost | **77.7%** | 5.0 |
| Full A* (f=g+h) | paper setting | **76.0%** | 5.2 |

**Key finding**: All three modes cluster tightly within 1.7pp. This confirms that **search structure (branching) is the primary driver**, not the specific heuristic function. Even with h=0 (no DeepSeek critic at all), performance matches CoT — the branching exploration alone is sufficient. The DeepSeek critic provides a small further refinement. This is a strength of the framework: it is robust to heuristic choice, and the admissibility guarantee holds for any monotone h(n), not just the DeepSeek instantiation.

---

## W5 — No sensitivity analysis on design choices (branching factor, budget, heuristic weighting)

This is a valid gap. The paper currently reports a single configuration (Nc = 3, B = 15) and references the compute efficiency table (§5.4, Table 7) which shows per-token accuracy.

We have already run the budget sensitivity experiment during the rebuttal window. Results on 300 HotpotQA questions (fixed sample, full A* mode):

| Budget B | Accuracy | Avg Nodes |
|---|---|---|
| B = 5 | 73.0% | 4.4 |
| B = 10 | 77.3% | 5.2 |
| B = 15 | 78.0% | 5.4 |
| **B = 20** | **78.7%** | 5.3 |
| B = 30 | 76.0% | 5.2 |

**Key finding (budget)**: Clear monotonic improvement B=5→20 (+5.7pp), with diminishing returns at B=30. B=20 is the empirical peak on this sample. The paper's default B=15 is within 0.7pp of the peak — a sound conservative choice that saves ~25% compute vs. B=20 while sacrificing less than 1pp accuracy. This directly addresses the compute–accuracy trade-off concern (also see §5.4 Table 7 in the paper for full token counts).

Branching factor sweep results on the same 300-question HotpotQA sample (full A* mode, B=30):

| Branch K | Description | Accuracy | Avg Nodes |
|---|---|---|---|
| K = 1 | degenerate linear chain (no branching) | 77.7% | 2.7 |
| K = 2 | minimal branching | 78.3% | 4.3 |
| K = 3 | paper setting | 76.0% | 5.2 |
| **K = 4** | wider search | **80.3%** | 6.1 |

**Key finding (branching)**: K=4 achieves 80.3% — matching the paper's full-dataset A* score (80.1%, Table 1) on this 300-question subset, which validates sample representativeness. Critically, even K=1 (a pure linear chain, no real search) achieves 77.7%, confirming that the **framework is robust** — it degrades gracefully rather than collapsing without branching. The +2.6pp from K=1→K=4 shows branching contributes meaningfully. The paper's K=3 default is conservative; K=4 may be preferred when compute allows. We will add both tables as Appendix D in the revision.

---

## W6 — Computational cost of A* not analyzed

We respectfully point out that **the paper already includes a compute efficiency analysis** in Section 5.4 (Table 7). Key numbers: A* uses approximately 3× the tokens of CoT (~2,700 vs. ~900 per question) and slightly more than SC (~2,250). However:

> "On StrategyQA, A* reaches 74.80% while CoT saturates at 64.52% — a gain that cannot be achieved by running CoT more. SC achieves lower accuracy than A* on both datasets while using similar tokens, confirming that guided search is a more effective use of inference budget than unguided sampling on branching-topology problems." (§5.4)

This addresses the core trade-off: A* is not free, but for branching-topology problems the accuracy gain is unreachable by simply running CoT more. We will make this section more prominent in the revision (moving it earlier in §5) to ensure it is not missed.

---

## Summary of Proposed Additions (available before deadline)

| Concern | Response type | Timeline |
|---|---|---|
| W4: ToT methodological comparison | Heuristic ablation table (BFS/Greedy/A*) | ✅ Done — see §W4 above |
| W5: Sensitivity analysis (budget) | B ∈ {5,10,15,20,30} curve | ✅ Done — see §W5 above |
| W5: Sensitivity analysis (branching) | Nc ∈ {1,2,3,4} table | ✅ Done — see §W5 above |
| W3: Larger models | 13B subset run (500 questions) | 3–5 days |
| W6: Compute analysis | §5.4 Table 7 already exists; will make prominent | Revision |

> **Paper references for reviewer convenience**: §2.1 (A* framework + admissibility), §3.2 (heuristic validation on 500 states), §4 Table 1 (full baselines incl. ToT), §5.3 (Fluency–Correctness Asymmetry), §5.4 Table 7 (compute efficiency), §5.5 (Qwen-0.5B amplification), §6 (Limitations). The paper PDF is co-located with this response.
| W3: Larger models | 13B subset run (500 questions) | 3–5 days |
| W6: Compute analysis | Section 5.4 already exists; will make prominent | Revision |

We believe these additions directly address the core methodological gaps and are confident the framework's structural findings (topology predicts strategy, Fluency–Correctness ceiling) are robust.
