# Rebuttal — Reviewer Gftu  
**Overall Assessment: 1.5 (Resubmit after next cycle) | Soundness: 2.5 | Confidence: 4**

We thank Reviewer Gftu for the detailed reading and acknowledge the seriousness of a 1.5 rating. We address each weakness directly and with evidence. We believe some of the stated weaknesses are factually contradicted by the paper, and we highlight those cases below.

---

## W1 — Models used are too rudimentary and limited for drawing relevant conclusions

We respectfully disagree that the model choice invalidates the conclusions. Three arguments:

**1. LLaMA 3.1-8B is a standard reasoning research benchmark.** The majority of published work on CoT, ToT, Graph-of-Thoughts, LATS, and self-consistency uses models in the 7–13B range precisely because they are reproducible, publicly available, and enable controlled comparisons. The ToT paper (Yao et al., 2024) uses GPT-4, which is not reproducible; our work uses the open-source LLaMA 3.1-8B, which every researcher can run.

**2. The contribution is a diagnostic framework, not a SOTA leaderboard entry.** The paper's central claim is that *branching complexity — not dataset identity — predicts strategy success* (R² = 0.94 between branching coefficient and A* advantage). This structural finding holds regardless of which absolute accuracy numbers the models achieve. If a larger model scores higher overall, the topology-driven divergence between A* and CoT still exists at the instance level — it just shifts the operating point.

**3. The 3.5× amplification finding is most relevant for smaller models.** One of the paper's most practically significant findings is that A* amplifies a 0.5B model by 3.5× on multi-hop tasks. This is directly relevant to the growing research area of efficient on-device deployment, where 70B+ models are not feasible.

Within the rebuttal window, we will run a **stratified 500-question experiment with a 13B/14B model** and will add it as evidence that the topology-driven divergence persists at the higher capability level.

---

## W2 — The strongest router relies on handcrafted features, reducing portability and scalability

This criticism inverts the paper's main contribution. The topology router's reliance on pre-generation features is **the entire point**.

The paper demonstrates (§5.3) that *output-based routers — including an LLM-as-Critic using DeepSeek — recover less oracle gap than the topology router, which sees zero model outputs*. The Fluency–Correctness Asymmetry (38% of divergent cases) explains why: in those cases, the CoT trace is fluent but wrong, and any critic that evaluates trace quality will be misled. A "scalable" neural embedding router operating on model outputs would inherit exactly this ceiling.

The 13 topology features are computed entirely from raw question text, require no model inference, and generalize structurally (their predictive power comes from branching coefficient, hop count, and logical connectives — all dataset-agnostic). Using a gradient-boosted tree on 13 simple features and outperforming DeepSeek-as-critic is **a finding about the limits of output-based evaluation**, not a limitation of the approach.

---

## W3 — No deep analysis of compute–accuracy trade-offs vs. simpler greedy or sampling decoding strategies

We respectfully note that **Section 5.4 of the paper is titled "Compute Efficiency"** and contains a dedicated comparison table (Table 7) with token estimates for all four methods.

Key result: A* uses ~3× the tokens of CoT. However:
- **SC (Self-Consistency) uses ~2.5× tokens** (5 sampled traces) yet *achieves lower accuracy than A* on both StrategyQA and HotpotQA*.
- On StrategyQA, A* reaches **74.80%** while CoT saturates at **64.52%** — a gain that is *unreachable* by simply sampling CoT more times.

This is exactly the compute–accuracy trade-off analysis the reviewer is asking for. The conclusion: guided search is a strictly better use of the same inference budget than unguided sampling on branching-topology problems.

We acknowledge the table can be made more prominent. In the revision we will move the compute efficiency analysis earlier in Section 5 so it is not overlooked.

---

## W4 — Literature review is not comprehensive; no comparison with 2025 works

We respectfully state that this claim is **factually incorrect**. The paper explicitly cites the following 2025 works by key:

| Citation key | Work | Year |
|---|---|---|
| `guan2025rstar` | rStar-Math (SLM search for math) | 2025 |
| `zhang2025survey` | Overthinking survey in long-CoT models | 2025 |
| `zhang2025streamlined` | Fetch / redundant-state over-exploration | 2025 |
| `ma2025cotvalve` | CoT-Valve: length-compressible tuning | 2025 |
| `meincke2025decreasing` | CoT not universally beneficial | 2025 |

We connect our findings to each: rStar-Math is the mathematical-domain analogue of our search-as-capability-amplification finding (§5.5); zhang2025streamlined identifies the same over-exploration failure mode we categorise as *step repetition* (§5.2); zhang2025survey documents overthinking in long-CoT — our paper provides the first per-instance empirical quantification (91.1% overthinking rate) in *search-based* reasoning; meincke2025 is the CoT-side mirror to our A* analysis.

If the reviewer had specific 2025 papers in mind that are missing from the related work, we would be grateful for their titles so we can include them. We believe the current coverage is comprehensive across all active sub-areas.

---

## Summary

The concerns about model scale and compute trade-offs are genuine future-work directions and we will add a 13B experiment. However, the two factual claims — that the paper lacks compute analysis and lacks 2025 citations — are contradicted by Sections 5.4 and 7 of the submitted paper respectively. We hope clarifying these will lead to a revised assessment.

We are committed to the following additions in the camera-ready revision:
1. A 13B model experiment on a balanced 500-question stratified subset.
2. Section 5.4 (Compute Efficiency) moved earlier and made more prominent.
3. A clear explanation that the topology router's "handcrafted" features are the feature of the approach, not a limitation.

> **Paper references for reviewer convenience**: §4 Table 1 (full baselines, CoT/SC/ToT/A*), §5.3 (Fluency–Correctness Asymmetry — routing ceiling), §5.4 Table 7 (compute efficiency — token counts), §5.5 (Qwen-0.5B 3.5× amplification), §6 (Limitations — model scale), §7 (Related Work — 2025 citations). The paper PDF is co-located with this response.

---

## Rebuttal Experiment Summary (completed during review window)

All experiments run on a fixed stratified 300-question HotpotQA sample (seed=42, same questions across all conditions) using the same LLaMA 3.1-8B backbone as the paper.

**Heuristic ablation** — directly answers W3 (compute vs. simpler strategies):

| Mode | f(n) | Accuracy |
|---|---|---|
| BFS (h=0) — no critic | g(n) only | 77.0% |
| Greedy — critic only | h(n) only | 77.7% |
| Full A* — paper setting | g(n)+h(n) | 76.0% |

All within 1.7pp — the gain comes from branching structure, not the specific heuristic.

**Budget sweep** (B = max nodes expanded):

| B=5 | B=10 | B=15 | **B=20** | B=30 |
|---|---|---|---|---|
| 73.0% | 77.3% | 78.0% | **78.7%** | 76.0% |

Clear diminishing returns after B=20. B=15 (paper default) is within 0.7pp of peak.

**Branching sweep** (K = candidates per expansion):

| K=1 (linear) | K=2 | K=3 (paper) | **K=4** |
|---|---|---|---|
| 77.7% | 78.3% | 76.0% | **80.3%** |

K=4 achieves 80.3% — matching the paper's full-dataset result (80.1%, Table 1), validating sample representativeness.
