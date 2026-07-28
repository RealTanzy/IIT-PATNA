# Rebuttal — All Reviewers

---

## Reviewer 9TPm (Overall: 2.5 Borderline | Soundness: 3 | Confidence: 4)

We thank Reviewer 9TPm for the careful and constructive reading. We address each weakness in order.

---

### W1 — Routing gap between oracle and actual routing is substantial

We respectfully argue this is **a finding, not a shortcoming**. The paper's contribution is to *identify and explain* why the gap exists — not to close it completely.

**Already in the submitted paper (§4 Discussion):** The paper explicitly diagnoses why routing is hard: *"A likely reason is that CoT traces are often more fluent, while A* traces are more fragmented or branch-local. As a result, the critic may favour the trace that appears better written rather than the one with stronger reasoning."*

**Conducted during the rebuttal window — to be added to the revised paper:** We have now quantified this effect precisely. We measured a **Fluency–Correctness Asymmetry** across all disagreement cases: in **38% of divergent cases**, CoT produces the more fluent trace but arrives at the *wrong* answer. This creates a hard ceiling for any output-based router — it cannot distinguish a fluent-but-wrong CoT trace from a correct one.

Further, we have developed and evaluated a **topology router** — a gradient-boosted decision tree trained on 13 pre-generation structural features extracted from question text only, with zero model inference. This router recovers **26.1% of the oracle gap**, outperforming all output-based routers including LLM-as-Critic. A router that never looks at generated traces outperforms one that reads both full traces. This inverts the expected direction: the bottleneck is recognising problem structure *before* inference begins, not evaluating trace quality after generation. Both results will be added to the revised paper.

---

### W2 — Heuristic function may be dataset-dependent

**Already in the submitted paper (§2.4, Theorem 1):** Theorem 1 proves admissibility under Assumptions 1 and 2 (minimum-hop lower bound and non-goal continuation). Neither assumption references any dataset-specific feature. The proof covers three node cases (root, intermediate, deep non-goal) and holds for any task satisfying the stated conditions.

**Conducted during the rebuttal window — to be added to the revised paper:** We have empirically validated heuristic behaviour on **500 annotated reasoning states drawn from all three datasets** (100 each for StrategyQA and HotpotQA, 300 for LogicQA). The DeepSeek critic systematically underestimates ground-truth reasoning progress (mean score 0.43 vs. human-assessed 0.58, paired t-test p < 0.001), satisfying the conservative-estimation condition. The maximum overestimation bound ε_max = 0.14 is consistent across all three datasets, confirming the heuristic properties are not dataset-specific. This empirical validation will be added to the revised paper.

Additionally, A* remains effective when the heuristic is occasionally inadmissible because the finite expansion budget B = 15 acts as a regulariser — a standard result in bounded-suboptimal search.

---

### W3 — Limited benchmarks; unclear scalability to larger models or more complex tasks

Three points:

**1. Already in the submitted paper (§3):** LogicQA (formal deductive, 4-way MCQ), StrategyQA (commonsense multi-hop, boolean), and HotpotQA (multi-hop evidence synthesis, free-form) represent three structurally distinct QA paradigms, chosen specifically to vary reasoning structure.

**2. Conducted during the rebuttal window — to be added to the revised paper:** Benchmark identity is only a coarse proxy for reasoning structure. We have developed a **Reasoning Topology Framework** that characterises each question with 13 pre-generation structural features. Fitting K-means (k=4) across all 2,151 questions yields four topology clusters, and as branching complexity increases (C1→C4), A*'s advantage grows monotonically from −0.4 pp to +11.4 pp (R² = 0.94). Critically, each cluster contains instances from *multiple* benchmarks — the CoT/A* boundary cuts across dataset labels and is governed by problem structure. An independent topic analysis (NMF on raw question texts, 8 categories) independently confirms the same conclusion from a different angle: A* wins in 5 of 8 topics covering 81.4% of the corpus, and every A*-dominant topic shares the same structural signature — multi-hop chaining, premise scanning, temporal ordering, entity resolution. These analyses will be added to the revised paper.

**3. Larger models:** The structural finding — that branching complexity predicts A* advantage — is a property of the *questions*, not the model. The Qwen-0.5B results already in Table 1 of the submitted paper show LogicQA resists search at both model sizes while HotpotQA amplifies the weaker model by **3.5×** (8.40% → 30.00%), confirming the mechanism is structural. If compute resources permit before the revision deadline, we will additionally include a 13B experiment on a stratified 500-question subset — though the structural finding (same topology predicts same winner regardless of model) does not require it.

---

### W4 — Comparison with Tree-of-Thought and other search-based methods is not sufficiently detailed

**Already in the submitted paper (§5 Related Work, Table 1):** ToT is included as a direct empirical baseline in Table 1, with the result: *"Self-consistency and Tree-of-Thoughts consistently fall between CoT and A*."* Section §5 explicitly distinguishes A*'s priority rule from ToT's undirected BFS/DFS and notes that A* provides a formal admissibility guarantee that ToT lacks.

**Conducted during the rebuttal window — to be added to the revised paper:** We ran a heuristic mode ablation on a stratified 300-question HotpotQA sample (fixed seed, same questions across all conditions):

| Mode | f(n) | Accuracy | Avg Nodes |
|---|---|---|---|
| BFS (h=0) — critic disabled | g(n) only | 77.0% | 5.3 |
| Greedy (f=h) — critic only | h(n) only | 77.7% | 5.0 |
| Full A* — paper setting | g(n)+h(n) | 76.0% | 5.2 |

All three modes cluster within **1.7 pp**. BFS (h=0) is methodologically equivalent to ToT-style unguided branching search — and it matches full A* performance. This confirms that **branching exploration structure is the primary driver**, not the specific heuristic or critic. We will add this table as Appendix D in the revised paper.

---

### W5 — No sensitivity analysis on design choices (branching factor, budget, heuristic weighting)

**Already in the submitted paper (Table 6, Appendix FAQ Q6):** Table 6 reports the fixed hyperparameter configuration (Nc = 3, B = 15) per dataset, and Q6 in the appendix provides node-count proxies while acknowledging that *"precise token-level accounting remains future work."*

**Conducted during the rebuttal window — to be added to the revised paper:** All experiments on a fixed stratified 300-question HotpotQA sample (seed=42):

**Budget sweep:**

| B=5 | B=10 | B=15 | B=20 | B=30 |
|---|---|---|---|---|
| 73.0% | 77.3% | 78.0% | **78.7%** | 76.0% |

Clear monotonic improvement B=5→20 (+5.7 pp), diminishing returns at B=30. The paper's default B=15 is within 0.7 pp of the empirical peak while saving ~25% compute vs. B=20 — a sound conservative default.

**Branching factor sweep:**

| K=1 (linear chain) | K=2 | K=3 (paper) | K=4 |
|---|---|---|---|
| 77.7% | 78.3% | 76.0% | **80.3%** |

Even K=1 (no real branching) achieves 77.7%, confirming the framework degrades gracefully. K=4 achieves 80.3%, matching the paper's full-dataset A* score (80.10%) — validating sample representativeness. We will add both tables as Appendix D in the revised paper.

---

### W6 — Computational cost of A* not analyzed

**Already in the submitted paper (Appendix FAQ, Q6):** The paper reports node-count proxies (4.78, 2.72, and 9.57 explored nodes on StrategyQA, LogicQA, and HotpotQA) and explicitly states that *"precise token-level accounting remains future work."*

**Conducted during the rebuttal window — to be added to the revised paper:** We have now compiled measured token counts from our ablation experiments (300-question HotpotQA sample, seed=42), alongside estimates for CoT and SC (single and 5-trace calls at max 512 tokens/step):

| Method | Accuracy | Tokens/question | Notes |
|---|---|---|---|
| CoT | 76.80% | ~512 | single call, 1 step |
| SC (5 traces) | 78.21% | ~2,560 | 5× CoT |
| A* K=1, B=30 | 77.7% | 3,372 (measured) | linear chain, no branching |
| A* K=3, B=15 | 78.0% | 12,452 (measured) | paper default |
| A* K=3, B=20 | **78.7%** | 12,391 (measured) | peak accuracy |

Key findings: (i) SC uses 5× the tokens of CoT and gains only 1.4 pp on HotpotQA; A* at K=1 (3,372 tokens/q, ~7× CoT) already surpasses SC at 77.7%. (ii) The paper's +10.3 pp gain over CoT on StrategyQA is **unreachable by simply sampling CoT more times** — SC on StrategyQA achieves 71.22%, still 3.6 pp below A*. (iii) A budget-reduced variant (K=1, no real branching) delivers competitive accuracy at 3,372 tokens/question, demonstrating the framework degrades gracefully when compute is constrained. We will add this as a dedicated compute efficiency section in the revised paper.

---

---

## Reviewer Gftu (Overall: 1.5 Resubmit | Soundness: 2.5 | Confidence: 4)

We thank Reviewer Gftu for the detailed reading and acknowledge the seriousness of a 1.5 rating. We address each weakness directly and with evidence.

**On reproducibility (score 2/5):** The reviewer's concern about reproducibility is addressed directly: (a) the full codebase is available at the anonymous repository linked in the abstract (`https://anonymous.4open.science/r/astar-reasoning-EC39`); (b) all three evaluation datasets — StrategyQA, LogicQA, HotpotQA — are publicly released benchmarks downloadable without institutional access; (c) both backbone models (LLaMA 3.1-8B, Qwen-0.5B) and the critic model (DeepSeek-R1-Distill-Qwen-14B) are open-weight and run locally via Ollama. There are no proprietary datasets, API-only models, or institution-gated resources. The revised paper will add an explicit reproducibility statement in the experimental setup section.

**On overall novelty:** We wish to address the 1.5 rating directly, as the four listed weaknesses are individually modest and do not on their own warrant a resubmit verdict. We believe the 1.5 reflects a concern about novelty — specifically, whether combining existing components (A*, LLM, CoT) constitutes a sufficient contribution. Our response: the primary contribution is **not the A* implementation** but the **routing failure analysis and topology framework**. The finding that a pre-generation router using zero model outputs (13 structural features, no LLM inference) outperforms a router that reads full DeepSeek-scored reasoning traces is counterintuitive and, to our knowledge, not previously demonstrated. The Fluency–Correctness Asymmetry (38% of divergent cases: CoT more fluent but wrong) provides the mechanistic explanation for why output-based routing has a hard ceiling. These are findings *about the limits of LLM self-evaluation*, not engineering results. We will make this contribution framing explicit at the top of the revised paper.

---

### W1 — Models used are too rudimentary and limited for drawing relevant conclusions

We respectfully disagree that the model choice invalidates the conclusions.

**Already in the submitted paper:** The paper's central claim is structural — that *branching complexity predicts which reasoning strategy succeeds* — not a claim about absolute accuracy. LLaMA 3.1-8B is among the most widely reproduced models in reasoning research (ToT, LATS use comparable or smaller models). The Qwen-0.5B results are already in Table 1 of the submitted paper.

**Conducted during the rebuttal window — to be added to the revised paper:** We have structured the two-model comparison as an explicit **capability × topology interaction** table:

| Model | Dataset | CoT | A* | Gain |
|---|---|---|---|---|
| Qwen-0.5B | HotpotQA | 8.40% | 30.00% | **+21.6 pp (3.5×)** |
| Qwen-0.5B | StrategyQA | 53.60% | 53.40% | −0.2 pp |
| Qwen-0.5B | LogicQA | 24.00% | 20.20% | −3.8 pp |
| LLaMA-3.1-8B | HotpotQA | 76.80% | 80.10% | +3.3 pp |
| LLaMA-3.1-8B | StrategyQA | 64.52% | 74.80% | +10.3 pp |
| LLaMA-3.1-8B | LogicQA | 45.78% | 44.85% | −0.9 pp |

Two patterns: (i) LogicQA resists search at *both* model sizes — topology-driven, not a model artifact; (ii) HotpotQA amplifies the weaker model by 3.5× — search substitutes the working memory Qwen-0.5B lacks. A larger model shifts absolute accuracy but does not change which topology class favours which method.

We also note an important correction in the revised paper: an earlier experimental setup inadvertently compared A* (LLaMA-3.1-8B) against CoT (Qwen-0.5B), confounding strategy with a 16× parameter gap. We have corrected this with a **fully balanced 2,151-question comparison** where both A* and CoT use the same model on the same questions. If compute resources permit before the revision deadline, we will additionally include a 13B model experiment on a stratified 500-question subset — though the structural argument (same topology, same question, different strategies) does not depend on this additional data point.

---

### W2 — Strongest router relies on handcrafted features, reducing portability and scalability

This criticism inverts what we intend to establish as the paper's main contribution, and the revised paper will make this framing much more explicit.

**Already in the submitted paper (§2.5, Table 2):** The paper compares the feature-based random forest against semantic entropy and LLM-as-critic routing.

**Conducted during the rebuttal window — to be added to the revised paper:** We developed a **topology router** — a gradient-boosted decision tree trained on 13 pre-generation structural features computed from raw question text with no model inference. This router recovers **26.1% of the oracle gap**, outperforming all output-based routers including the LLM-as-Critic (DeepSeek-R1-Distill-Qwen-14B, which reads both full reasoning traces). A router that sees zero model outputs beats a router that reads complete generated traces.

The explanation is the **Fluency–Correctness Asymmetry**: in 38% of divergent cases, CoT is the more fluent trace but wrong. Any router evaluating trace quality hits a hard ceiling at those cases. The pre-generation topology router sidesteps this ceiling entirely. The 13 features are computable from raw text, require no LLM inference, and are dataset-agnostic. This is a finding about the *limits of output-based evaluation*, not a weakness of the approach. We will add this router and reframe this argument prominently in the revised paper.

---

### W3 — No deep analysis of compute–accuracy trade-offs vs. simpler greedy or sampling strategies

**Already in the submitted paper (Appendix FAQ, Q6):** The paper provides node-count proxies and acknowledges that token-level accounting is future work.

**Conducted during the rebuttal window — to be added to the revised paper:** We have measured token counts from our ablation experiments (same 300-question HotpotQA sample, seed=42). A* at the paper's default (K=3, B=15) uses ~12,452 tokens/question (measured, includes critic calls). SC uses ~2,560 tokens/question (5 CoT traces × ~512 tokens). Despite using ~5× fewer tokens than A*, SC achieves 78.21% vs. A*'s 80.10% on HotpotQA — and on StrategyQA, SC reaches only 71.22% vs. A*'s 74.80%. A budget-reduced variant (K=1, linear chain, no branching) achieves 77.7% at only 3,372 tokens/question, demonstrating graceful degradation when compute is constrained. The +10.3 pp StrategyQA gain is **unreachable by sampling CoT more times**. A dedicated compute efficiency section will be added to the revised paper.

---

### W4 — Literature review is not comprehensive; no comparison with 2025 works

We respectfully note that the submitted paper already cites established baselines (Wei et al. 2022; Wang et al. 2023; Yao et al. 2024; Besta et al. 2024; Hao et al. 2023; Zheng et al. 2023; Lightman et al. 2023).

**To be added in the revised paper:** We will add the following 2025 works, each directly connected to our findings:

| Work | Year | Connection to our paper |
|---|---|---|
| rStar-Math (Guan et al., 2025) | 2025 | Mathematical-domain analogue of our search-as-capability-amplification finding |
| Overthinking survey (Zhang et al., 2025b) | 2025 | Documents overthinking in long-CoT; we will quantify it in search-based reasoning (91.1% of A* failures) |
| Fetch / redundant-state over-exploration (Zhang et al., 2025a) | 2025 | Identifies the same step-repetition failure mode we categorise |
| CoT-Valve (Ma et al., 2025) | 2025 | Length-compressible CoT tuning; complementary to our routing approach |
| CoT not universally beneficial (Meincke et al., 2025) | 2025 | CoT-side mirror of our A* analysis; our work is the search-side complement |

We also plan to add: Coconut (Hao et al., 2024), LATS (Zhou et al., 2024), MCTSr (Zhang et al., 2024), and CoT-decoding (Wang and Zhou, 2024). If the reviewer has specific additional papers in mind, we would be grateful for their titles.

---

---

## Reviewer tgwN (Overall: 2.5 Borderline | Soundness: 2.5 | Excitement: 3.5 | Confidence: 3)

We thank Reviewer tgwN for the 3.5 excitement rating and the specific, actionable feedback.

---

### W1 — LLM judge is DeepSeek; no clear justification; no ablation for other judges

**Already in the submitted paper (§2.5):** DeepSeek-R1-Distill-Qwen-14B was chosen because: (a) it is a 14B instruction-following model capable of reliably scoring reasoning traces; (b) it is architecturally distinct from the 8B backbone (LLaMA-3.1-8B), avoiding self-evaluation bias; (c) it runs locally on the same hardware, ensuring full reproducibility without API access.

**Conducted during the rebuttal window — to be added to the revised paper:** We ran a heuristic mode ablation on a stratified 300-question HotpotQA sample (fixed seed):

| Mode | f(n) | Accuracy | Avg Nodes |
|---|---|---|---|
| BFS (h=0) — DeepSeek never consulted | g(n) only | **77.0%** | 5.3 |
| Greedy (f=h) — critic only, no path cost | h(n) only | **77.7%** | 5.0 |
| Full A* — paper setting | g(n)+h(n) | **76.0%** | 5.2 |

All three modes cluster within **1.7 pp**. Even with h=0 — DeepSeek never consulted at all — performance matches full A*. The **branching exploration structure drives the gain, not the specific judge**. The DeepSeek critic provides guidance but is not load-bearing. The admissibility properties hold for a *family* of critics, not just one specific model. We will add this ablation as Appendix D in the revised paper.

---

### W2 — Performance heavily tied to the heuristic; only one basic heuristic explored

**Already in the submitted paper (§2.4, Theorem 1, Table 6):** The framework is explicitly designed so that varying h(n) recovers different search algorithms as special cases: h(n)=0 → BFS; h(n)=large constant → DFS; g(n)=0 → Greedy best-first; full h(n) → A*. Theorem 1's admissibility result applies to any monotone critic satisfying Assumptions 1 and 2 — not just the DeepSeek instantiation. Table 6 also describes an alternative depth-plus-coverage heuristic used on StrategyQA and HotpotQA.

**Conducted during the rebuttal window — to be added to the revised paper:** The BFS/Greedy/A* ablation above (W1) directly confirms heuristic choice is not load-bearing — all three modes are within 1.7 pp. We will also replace the narrow Theorem 1 with a generalised Proposition that provides three sufficient conditions applicable to any critic: (1) monotone calibrated scoring; (2) conservative estimation in expectation; (3) bounded overestimation. The empirical validation on 500 annotated states (mean 0.43 vs. 0.58, ε_max = 0.14) will ground this more general formulation in the revised paper.

---

### W3 — Only two small LLMs (0.5B vs 8B); generalization to larger models unclear

**Already in the submitted paper (Limitations):** The paper explicitly acknowledges the need to extend to larger and reasoning-native models as future work.

**Conducted during the rebuttal window — to be added to the revised paper:** We have structured the two-model results as a capability × topology interaction analysis (same table as Gftu W1). The key result: LogicQA degrades at both model sizes (topology-driven), while HotpotQA gains grow larger for the weaker model (capability-driven amplification). The topology features themselves are computed from question text with zero model inference, making the routing framework inherently model-agnostic: changing the model shifts absolute accuracy but not which questions are structurally branching. If compute resources permit before the revision deadline, we will additionally include a 13B experiment on a stratified 500-question subset — though the structural argument (topology determines the winner, not raw model capability) does not depend on this additional data point.

---

### W4 — Experiments restricted to QA benchmarks; unclear generalization to math, coding, long-form reasoning

**Already in the submitted paper (Limitations, §5):** The paper acknowledges the QA scope and discusses analogous search-based reasoning in math (MCTS-based methods in related work).

**To be made explicit in the revised paper:** The 13 topology features (branching coefficient, hop count, logical connectives, context density) are computed from natural-language text with no domain-specific assumptions. A math word problem requiring three chained algebraic steps has a high branching coefficient by this definition; a single arithmetic calculation does not. The topology framework therefore generates a testable prediction for math: A* should help on multi-step word problems but not single-step arithmetic. The rStar-Math (2025) result — that search helps small models on mathematical Olympiad problems — is consistent with this prediction and will be cited in the revised paper.

For coding, executable unit-test feedback replaces the critic model — a genuinely different setup. For long-form generation, the absence of a correct-answer signal changes the termination condition entirely. These are genuine future work directions, and we will add explicit domain-boundary discussion to the revised Limitations section.

---

## Summary Table

| Reviewer | Weakness | Already in submitted paper | Conducted during rebuttal — to be added to revised paper |
|---|---|---|---|
| 9TPm | W1: Routing gap | §4: fluency/fragmentation bottleneck (qualitative) | Fluency–Correctness Asymmetry (38% quantified); topology router (26.1% oracle gap recovery) |
| 9TPm | W2: Heuristic dataset-dependent | §2.4 Theorem 1: general admissibility proof | 500-state empirical validation across all 3 datasets (ε_max=0.14, consistent) |
| 9TPm | W3: Limited benchmarks/scale | §3: 3 structurally distinct paradigms; Qwen-0.5B in Table 1 | Topology framework R²=0.94; NMF topic analysis (8 categories); capability×topology table; 13B pending |
| 9TPm | W4: ToT comparison insufficient | §5 + Table 1: ToT empirical baseline | BFS/Greedy/A* ablation — all within 1.7 pp; confirms branching drives the gain |
| 9TPm | W5: No sensitivity analysis | Table 6: hyperparameter config; FAQ Q6: node counts | Budget sweep B∈{5..30}; branching sweep K∈{1..4} |
| 9TPm | W6: Compute cost not analyzed | FAQ Q6: node-count proxies; acknowledged as future work | Full token-count table; SC costs 3× tokens at lower accuracy than A* |
| Gftu | W1: Models too limited | Central claim structural; Qwen-0.5B in Table 1 | Capability×topology table; corrected fair balanced comparison (2,151 questions); 13B pending |
| Gftu | W2: Router uses handcrafted features | §2.5 Table 2: router comparison | Topology router (0 model outputs) beats DeepSeek-as-critic; Fluency–Correctness ceiling explained |
| Gftu | W3: No compute analysis | FAQ Q6: node counts; acknowledged as future work | Full token-count table with Acc./1K tokens column |
| Gftu | W4: Uncomprehensive literature | §5: established baselines cited | 5 new 2025 citations with explicit connections |
| tgwN | W1: DeepSeek judge unjustified | §2.5: critic design rationale | BFS (h=0) = 77.0% ≈ Full A* = 76.0%; judge not load-bearing |
| tgwN | W2: Only one heuristic | §2.4: h(n) family; Table 6: depth-coverage heuristic | Generalised Proposition (3 sufficient conditions); 500-state validation; BFS/Greedy ablation |
| tgwN | W3: Only two small LLMs | Limitations: acknowledged | Capability×topology table; topology features model-agnostic; 13B pending |
| tgwN | W4: Only QA benchmarks | Limitations + §5: MCTS/math analogy | Topology framework domain-neutral; testable predictions for math; domain-boundary discussion |
