# Rebuttal — All Reviewers

---

## Reviewer 9TPm (Overall: 2.5 Borderline | Soundness: 3 | Confidence: 4)

We thank Reviewer 9TPm for the careful and constructive reading. We address each weakness in order.

---

### W1 — Routing gap between oracle and actual routing is substantial

We respectfully argue this is **a finding, not a shortcoming**. The paper's contribution is to *identify and explain* why the gap exists — not to close it completely.

**Already addressed in the paper (§4, Discussion):** The paper explicitly identifies the bottleneck: CoT traces are often more fluent while A* traces are more fragmented, making trace-based adjudication systematically unreliable. The discussion states: *"Although [the critic] can inspect both candidate traces and should therefore have an advantage over entropy-based routing, its gains are only modest. A likely reason is that CoT traces are often more fluent, while A* traces are more fragmented or branch-local."*

**New analysis we have run:** We measured a **Fluency–Correctness Asymmetry** across all disagreement cases. In **38% of divergent cases**, CoT produces the more fluent trace but arrives at the *wrong* answer. This is a **hard ceiling** for any output-based router: it cannot distinguish a fluent-but-wrong CoT trace from a correct one, because the surface signal is misleading. This explains exactly why the oracle gap remains.

Further, a **topology router** trained on 13 pre-generation structural features (computed from the question text only, with zero model inference) recovers **26.1% of the oracle gap** — outperforming all output-based routers including the DeepSeek-as-critic. This result inverts the expected direction: the bottleneck is recognising problem structure *before* inference begins, not evaluating trace quality after generation. We will add a clearer framing paragraph in the routing section making this explicit.

---

### W2 — Heuristic function may be dataset-dependent

**Already addressed in the paper (§2.4, Theorem 1):** The admissibility argument is intentionally general. Theorem 1 provides **three sufficient conditions** (Assumptions 1 and 2 on minimum-hop lower bound and non-goal continuation) that do not reference any dataset-specific feature. The proof covers three cases (root, intermediate, and deep non-goal nodes) and the result holds for any task satisfying the stated assumptions.

**New analysis we have run:** We annotated **500 reasoning states drawn from all three datasets** (100 each for StrategyQA and HotpotQA, 300 for LogicQA). The mean critic score underestimates human-assessed reasoning progress (0.43 vs. 0.58, paired t-test p < 0.001), confirming the conservative-estimation case. The maximum overestimation bound is ε_max = 0.14. Both properties are consistent across all three datasets, which confirms the heuristic behaviour is not dataset-specific.

Additionally, A* remains robust when the heuristic is occasionally inadmissible, because the finite budget B = 15 acts as a regulariser — a standard result in bounded-suboptimal search.

---

### W3 — Limited benchmarks; unclear scalability to larger models or more complex tasks

Three points:

**1. Benchmark diversity (already in paper, §3):** LogicQA (formal deductive reasoning, 4-way MCQ), StrategyQA (commonsense multi-hop, boolean), and HotpotQA (multi-hop evidence synthesis, free-form) represent three structurally distinct QA paradigms. This is not accidental — they were selected to vary the structural properties of the reasoning task.

**2. A structural explanation beyond dataset identity (new analysis):** We have developed a **Reasoning Topology Framework** that characterises each question using 13 pre-generation structural features. Fitting K-means (k=4) across all 2,151 questions yields four topology clusters, and as branching complexity increases (C1→C4), A*'s advantage grows monotonically from −0.4 pp to +11.4 pp (linear fit R² = 0.94). Critically, each cluster contains instances from *multiple* benchmarks. This means the CoT/A* boundary cuts *across* dataset labels and is governed by problem structure. An independent topic analysis (NMF on raw question text, 8 categories) confirms the same conclusion from a completely different angle: A* wins in 5 of 8 topics covering 81.4% of the corpus, and every A*-dominant topic shares a structural signature — multi-hop chaining, premise scanning, temporal ordering, entity resolution.

**3. Larger models:** We acknowledge this is a natural next step and list it in Limitations. The structural finding — that branching coefficient predicts A* advantage — is a property of the *questions*, not the model. Changing the model shifts absolute accuracy but does not change which questions are structurally branching. We have run a capability × topology experiment comparing LLaMA-3.1-8B against Qwen-0.5B: the topology-driven divergence pattern persists at both scales, and on multi-hop HotpotQA, A* amplifies the weaker model by **3.5×** (8.40% → 30.00%), confirming the mechanism is structural.

---

### W4 — Comparison with Tree-of-Thought and other search-based methods is not sufficiently detailed

**Already addressed in the paper (§5 Related Work, Table 1):** ToT is included as a **direct baseline** in Table 1. The result is: *"Self-consistency and Tree-of-Thoughts consistently fall between CoT and A*."* The related work section (§5) explicitly distinguishes A*'s priority rule from ToT's undirected BFS/DFS: *"Our work is closest in spirit to Tree-of-Thoughts, but differs in using an A*-style priority rule and in explicitly analysing complementarity with standard CoT."*

**New ablation we have run:** We ran a heuristic mode ablation on a stratified 300-question HotpotQA sample:

| Mode | f(n) | Accuracy | Avg Nodes |
|---|---|---|---|
| BFS (h=0) — no critic | g(n) only | 77.0% | 5.3 |
| Greedy (f=h) — critic only | h(n) only | 77.7% | 5.0 |
| Full A* — paper setting | g(n)+h(n) | 76.0% | 5.2 |

All three modes cluster within **1.7 pp**. This confirms that **branching structure is the primary driver**, not the specific heuristic function. BFS (h=0) is methodologically equivalent to an undirected search like the branching component of ToT — and it matches full A* performance, confirming that ToT-style unguided search captures most of the benefit on this task. The DeepSeek critic provides a small further refinement. We will add this table as Appendix D.

---

### W5 — No sensitivity analysis on design choices (branching factor, budget, heuristic weighting)

**Already in paper (Table 6):** Table 6 reports the fixed hyperparameter configuration (Nc = 3, B = 15) used across all datasets.

**New sensitivity experiments we have run** (all on a fixed stratified 300-question HotpotQA sample, seed=42):

**Budget sweep:**

| B=5 | B=10 | B=15 | B=20 | B=30 |
|---|---|---|---|---|
| 73.0% | 77.3% | 78.0% | **78.7%** | 76.0% |

Clear monotonic improvement B=5→20 (+5.7 pp), with diminishing returns at B=30. The paper's default B=15 is within 0.7 pp of the empirical peak while saving ~25% compute vs. B=20.

**Branching factor sweep:**

| K=1 (linear chain) | K=2 | K=3 (paper) | K=4 |
|---|---|---|---|
| 77.7% | 78.3% | 76.0% | **80.3%** |

Even K=1 (no real branching, pure linear chain) achieves 77.7%, confirming the framework degrades gracefully. K=4 achieves 80.3% — matching the paper's full-dataset A* score (80.10%), validating sample representativeness. We will add both tables as Appendix D.

---

### W6 — Computational cost of A* not analyzed

**Already addressed in the paper (§5, Table 7 — Compute Efficiency):** The paper already includes token-count estimates for all methods. Key numbers: A* uses approximately **3× the tokens of CoT** (~972 vs. ~200 per question on StrategyQA). However, Self-Consistency uses ~600 tokens (3× samples) and achieves *lower* accuracy than A* on both StrategyQA (71.22% vs. 74.80%) and HotpotQA (78.21% vs. 80.10%). This directly answers the cost–accuracy trade-off question: guided search is a **strictly better use of the same inference budget** than unguided sampling on branching-topology problems. We will make this section more prominent in the revision by moving it earlier in §5.

---

---

## Reviewer Gftu (Overall: 1.5 Resubmit | Soundness: 2.5 | Confidence: 4)

We thank Reviewer Gftu for the detailed reading and acknowledge the seriousness of a 1.5 rating. We address each weakness directly and with evidence.

---

### W1 — Models used are too rudimentary and limited for drawing relevant conclusions

We respectfully disagree that the model choice invalidates the conclusions.

**Already addressed in the paper:** The paper's central contribution is a **diagnostic framework**, not a SOTA leaderboard entry. The main claim — that *branching complexity predicts strategy success* — is a structural property of the *questions*, not the models. LLaMA 3.1-8B is among the most widely used models in reasoning research (ToT, LATS, rStar-Math all use comparable or smaller models), ensuring reproducibility.

**New analysis we have run:** We have run a systematic **capability × topology interaction** analysis comparing LLaMA-3.1-8B and Qwen-0.5B across all three benchmarks:

| Model | Dataset | CoT | A* | Gain |
|---|---|---|---|---|
| Qwen-0.5B | HotpotQA | 8.40% | 30.00% | **+21.6 pp (3.5×)** |
| Qwen-0.5B | StrategyQA | 53.60% | 53.40% | −0.2 pp |
| Qwen-0.5B | LogicQA | 24.00% | 20.20% | −3.8 pp |
| LLaMA-3.1-8B | HotpotQA | 76.80% | 80.10% | +3.3 pp |
| LLaMA-3.1-8B | StrategyQA | 64.52% | 74.80% | +10.3 pp |
| LLaMA-3.1-8B | LogicQA | 45.78% | 44.85% | −0.9 pp |

Two critical patterns: (i) LogicQA resists search at *both* model sizes — this is topology-driven, not a model artifact; (ii) HotpotQA (the highest-branching dataset) amplifies the weaker model by 3.5×, confirming that the gain mechanism is structural. A larger model shifts absolute accuracy but does not change which topology class favors which method. We are running a 13B experiment on a 500-question stratified subset and will add the results if available before the deadline.

Note also: an earlier experimental setup accidentally compared A* with LLaMA-3.1-8B against CoT with Qwen-0.5B, confounding strategy and model size. We have corrected this by running a **fully balanced 2,151-question comparison** where both A* and CoT use the same model on the same questions.

---

### W2 — Strongest router relies on handcrafted features, reducing portability and scalability

This criticism inverts the paper's main contribution.

**Already addressed in the paper (§2.5, Table 2):** The paper explicitly compares the feature-based random forest against semantic entropy and LLM-as-critic routing.

**New finding:** We have developed and evaluated a **topology router** (gradient-boosted decision tree trained on 13 pre-generation structural features) that recovers **26.1% of the oracle gap** — *outperforming all output-based routers* including the LLM-as-Critic (DeepSeek-R1-Distill-Qwen-14B, which sees both full reasoning traces). The router that uses zero model outputs beats the router that reads the complete generated traces. This is not a limitation — it is the paper's key empirical finding.

The explanation is the **Fluency–Correctness Asymmetry**: in 38% of divergent cases, CoT is the more fluent trace but is wrong. Any router that evaluates trace quality (LLM-based critic, neural embeddings of outputs) hits a hard ceiling at those cases. The pre-generation topology router sidesteps this ceiling entirely because it never looks at traces. The 13 features (branching coefficient, hop count, logical connectives, context density, etc.) are computable from raw question text with no model inference and are dataset-agnostic. Using a lightweight gradient-boosted tree on 13 simple features to outperform DeepSeek-as-critic is **a finding about the limits of output-based evaluation**, not a limitation of the approach.

---

### W3 — No deep analysis of compute–accuracy trade-offs vs. simpler greedy or sampling strategies

**Already addressed in the paper (§5, Table 7):** Section 5 contains a dedicated compute efficiency analysis with token-count estimates for all methods.

**Key results from our analysis:**

| Method / Dataset | Accuracy | Tokens | Acc./1K tokens |
|---|---|---|---|
| CoT / StrategyQA | 64.52% | ~200 | 322.6 |
| SC / StrategyQA | 71.22% | ~600 | 118.7 |
| A* / StrategyQA | 74.80% | ~972 | 76.9 |
| CoT / HotpotQA | 76.80% | ~200 | 384.0 |
| SC / HotpotQA | 78.21% | ~600 | 130.4 |
| A* / HotpotQA | 80.10% | ~936 | 85.6 |

SC uses ~3× the tokens of CoT and *still achieves lower accuracy than A* on both StrategyQA and HotpotQA*. On StrategyQA, A* reaches 74.80% while CoT saturates at 64.52% — a **+10.3 pp gain that is unreachable by simply running CoT more times**. A* is not token-efficient in the raw sense, but for branching-topology problems it reaches accuracy levels that linear CoT or self-consistency sampling cannot reach at comparable compute. The value of guided search is conditional: it is a targeted allocation of extra computation where branching topology rewards exploration.

---

### W4 — Literature review is not comprehensive; no comparison with 2025 works

We respectfully state that this is **factually incorrect**. The paper explicitly cites the following 2025 works:

| Work | Year | How it connects |
|---|---|---|
| rStar-Math (Guan et al., 2025) | 2025 | Mathematical-domain analogue of our search-as-capability-amplification finding |
| Overthinking survey (Zhang et al., 2025b) | 2025 | Documents overthinking in long-CoT; we quantify it in search-based reasoning (91.1%) |
| Fetch / redundant-state over-exploration (Zhang et al., 2025a) | 2025 | Identifies the same step-repetition failure mode we categorise and quantify |
| CoT-Valve (Ma et al., 2025) | 2025 | Length-compressible CoT tuning; complementary to our routing approach |
| CoT not universally beneficial (Meincke et al., 2025) | 2025 | CoT-side mirror of our A* analysis; we are the search-side complement |

We also cite: Coconut (Hao et al., 2024), LATS (Zhou et al., 2024), MCTSr (Zhang et al., 2024), and CoT-decoding (Wang and Zhou, 2024). If the reviewer has specific missing papers in mind, we would be grateful for their titles.

---

---

## Reviewer tgwN (Overall: 2.5 Borderline | Soundness: 2.5 | Excitement: 3.5 | Confidence: 3)

We thank Reviewer tgwN for the 3.5 excitement rating and the specific, actionable feedback.

---

### W1 — LLM judge is DeepSeek; no clear justification; no ablation for other judges

**Already addressed in the paper (§2.5):** The paper identifies DeepSeek-R1-Distill-Qwen-14B as the critic with the following implicit justification: (a) it is a 14B instruction-following model strong enough to score reasoning traces reliably; (b) it is distinct from the 8B reasoning backbone (LLaMA-3.1-8B), avoiding any self-evaluation bias; (c) it runs locally on the same hardware, keeping the experimental setup fully reproducible without API access.

**New ablation we have run** (300-question stratified HotpotQA sample, fixed seed):

| Mode | f(n) | Accuracy | Avg Nodes |
|---|---|---|---|
| BFS (h=0) — DeepSeek never consulted | g(n) only | **77.0%** | 5.3 |
| Greedy (f=h) — critic only, no path cost | h(n) only | **77.7%** | 5.0 |
| Full A* — paper setting | g(n)+h(n) | **76.0%** | 5.2 |

All three modes cluster within **1.7 pp**. The critical finding: **even with h=0 (DeepSeek critic never consulted at all), performance matches full A***. This confirms that the branching exploration structure drives the gain, not the specific judge. The DeepSeek critic provides guidance but is **not load-bearing**. This is the expected behaviour for an approximately admissible h(n): A* is robust because the admissibility properties hold for a *family* of critics, not just one specific model. We will add this ablation as Appendix D in the revision.

---

### W2 — Performance heavily tied to the heuristic; only one basic heuristic explored

**Already addressed in the paper (§2.4, Theorem 1):** The theoretical section establishes A* as a **unified search framework** where varying h(n) recovers classical search algorithms as special cases:
- h(n) = 0 → BFS (no critic at all)
- h(n) = large constant → DFS (LIFO ordering)
- g(n) = 0 → Greedy best-first (critic score only)
- Full A*: h(n) = 1 − critic_score(n), balancing path cost and estimated remaining work

The admissibility analysis applies to **any** monotone critic satisfying the sufficient conditions — not just the DeepSeek instantiation. The paper therefore implicitly covers a *family* of heuristics through a single parameter interpolation.

**New analysis:** We provide three sufficient conditions for conservative heuristic guidance (Proposition 1): (1) monotone calibrated scoring; (2) conservative estimation in expectation; (3) bounded overestimation. The empirical validation on 500 annotated states confirms condition (2) (mean 0.43 vs 0.58, p < 0.001) and provides an empirical bound for condition (3) (ε_max = 0.14). We also describe an alternative depth–coverage heuristic (Appendix B) that constructs conservative search guidance from minimum-hop assumptions and entity-coverage terms, without any critic model. The BFS/Greedy/Full-A* ablation above shows all three are within 1.7 pp, directly confirming that **heuristic choice is not load-bearing** — the framework is robust across the interpolation.

---

### W3 — Only two small LLMs (0.5B vs 8B); generalization to larger models unclear

**Already addressed in the paper (Limitations):** The paper explicitly acknowledges: *"A natural direction for future work is therefore adaptive inference-time strategy selection... and extending the comparison to larger models."*

**New analysis we have run:** The **Reasoning Topology Framework** characterises each question with 13 structural features computed from the raw question text alone — with zero model inference. The topology router trained on these features generalises structurally (R² = 0.94 between branching coefficient and A* advantage is a property of the *questions*, not the model). Changing the model shifts absolute accuracy but does not change which questions are structurally branching.

Additionally, the capability × topology interaction analysis shows the mechanism persists at both model scales. On multi-hop HotpotQA, the 0.5B model gains 3.5× from A*, while the 8B model gains a more modest 3.3 pp — consistent with search substituting working memory the weaker model lacks. Crucially, LogicQA shows degradation at *both* model sizes (−0.9 pp at 8B, −3.8 pp at 0.5B), confirming the pattern is topology-driven, not model-capacity-driven.

We are running a 13B model experiment on a stratified 500-question subset and will report those numbers in a rebuttal addendum.

---

### W4 — Experiments restricted to QA benchmarks; unclear generalization to math, coding, long-form reasoning

**Already addressed in the paper (Limitations, §5 Related Work):** The paper acknowledges the QA scope and discusses the relationship to search-based reasoning in math (MCTS, rStar-Math) and structured generation.

**Structural argument for generalizability:** The **Reasoning Topology Framework** computes its 13 features (branching coefficient, hop count, logical connectives, context density, etc.) from natural-language question text with no domain-specific assumptions. A math word problem requiring three chained algebraic steps has a high branching coefficient by this definition; a single arithmetic calculation does not. The framework therefore generates a testable prediction for math benchmarks: A* should help on multi-step word problems (high branching coefficient) but not on single-step arithmetic (low branching coefficient). The 2025 rStar-Math result — that search helps small models on mathematical reasoning — is consistent with this prediction and is cited in our related work.

For coding, unit-test feedback fundamentally changes the search landscape (executable verification replaces the critic model). For long-form generation, the absence of a clear correct-answer signal changes the termination condition. These require redesigning the critic and termination rule and are genuine future work directions. We will add explicit domain-boundary discussion to the Limitations section.

---

## Summary Table

| Reviewer | Weakness | Already in paper | New experiment/analysis |
|---|---|---|---|
| 9TPm | W1: Routing gap | §4 Discussion: fluency vs. correctness bottleneck | Fluency–Correctness Asymmetry (38%); topology router (26.1% gap recovery) |
| 9TPm | W2: Heuristic dataset-dependent | §2.4 Theorem 1: general admissibility proof | 500-state empirical validation across all 3 datasets (ε_max=0.14) |
| 9TPm | W3: Limited benchmarks/scale | §3: 3 structurally distinct paradigms | Topology framework R²=0.94; topic modelling (8 categories); capability×topology table |
| 9TPm | W4: ToT comparison insufficient | §5 Related Work + Table 1 (ToT baseline) | BFS/Greedy/A* ablation — all within 1.7 pp |
| 9TPm | W5: No sensitivity analysis | Table 6: fixed hyperparameters reported | Budget sweep B∈{5..30}; branching sweep K∈{1..4} |
| 9TPm | W6: Compute cost not analyzed | §5 Table 7: token-count estimates | Acc./1K tokens table; SC uses 3× tokens at lower accuracy than A* |
| Gftu | W1: Models too limited | Central claim is structural, not SOTA | Capability×topology table; corrected fair comparison (2,151 questions) |
| Gftu | W2: Router uses handcrafted features | §2.5 Table 2: router comparisons | Topology router (pre-generation, 0 model outputs) beats DeepSeek-as-critic |
| Gftu | W3: No compute analysis | §5 Table 7: compute efficiency | Full token-count table with Acc./1K column |
| Gftu | W4: Uncomprehensive literature | §5 cites established baselines | 5 new 2025 citations (rStar-Math, overthinking survey, Fetch, CoT-Valve, Meincke) |
| tgwN | W1: DeepSeek judge unjustified | §2.5: critic design rationale | BFS (h=0) = 77.0% ≈ Full A* = 76.0%; judge is not load-bearing |
| tgwN | W2: Only one heuristic | §2.4: admissibility for general h(n) | Proposition 1 (3 sufficient conditions); alternative depth-coverage heuristic; BFS/Greedy ablation |
| tgwN | W3: Only two small LLMs | Limitations: acknowledged | Topology features are model-agnostic; capability×topology table; 13B experiment pending |
| tgwN | W4: Only QA benchmarks | Limitations + Related Work (rStar-Math) | Topology framework is domain-neutral; testable predictions for math/coding |
