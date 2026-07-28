# The Best Is Not Always Best
### Stakeholder Presentation — April 2026

---

## Slide 1 — Why We Restructured the Paper

### The Problem with the Original Structure
| Issue | What Was Wrong |
|---|---|
| Weak motivation | The intro presented A* as a solution, not as a paradox — burying the core finding |
| Disconnected sections | Methods and results were not causally linked; reader couldn't follow *why* each experiment existed |
| Missing narrative | No clear thread: "Here is the problem → here is what we discovered → here is what it means" |
| Contribution scatter | 6 contributions buried mid-paper; none foregrounded as the organising insight |
| No diagnostic framing | The paper read as "we tried A*, here are numbers" — not "A* fails in a predictable way and here's why" |

### What We Changed
- **Intro reframed** as a paradox: *A* is provably optimal — yet it fails 35%+ of the time. Why?*
- **New §2.5** added: Problem Structure Feature Extraction (13 topology features) — defines the *why* before results
- **New §3.2** Topology Clustering: empirically validates the framework with real data (744 paired items, K-means)
- **New §3.3** Topology-Aware Router: shows pre-generation routing beats post-generation routing
- **New §3.7** Efficiency Analysis: justifies A*'s token cost
- **Section order fixed**: Problem → Formal Method → Feature Framework → Results → Insights → Conclusion

---

## Slide 2 — Updated Paper at a Glance

### Title
**"The Best Is Not Always Best: When A* Search Fails to Improve LLM Reasoning and Why"**

### Clean Section Map (Final Paper — 15 pages)
```
§1 Introduction        → The paradox + 6 foregrounded contributions
§2 Methodology         → CoT, A*, topology features, routing strategies
§3 Results & Analysis  → 7 sub-sections, each with a clear finding
§4 Discussion          → Why routing is topology recognition, not trace comparison
§5 Related Work        → Positioned against ToT, MCTS, stop-overthinking literature
§6 Conclusion          → One-line thesis + open problem
Appendix               → Human eval, qualitative examples, failure taxonomy, FAQs
```

### Key Formatting Fixes
- All tables have captions, labels, and are cross-referenced
- Formal theorem (admissibility) is numbered and proved inline
- Consistent notation across all equations
- PDF: **255 KB, 15 pages, zero LaTeX errors**

---

## Slide 3 — Experimental Summary

| # | Experiment | Objective | Method | Result | Failed? | Why It Failed / Limitation |
|---|---|---|---|---|---|---|
| 1 | **Main Accuracy** | Compare CoT vs A* across 3 benchmarks | LLaMA-3.1-8B on StrategyQA, LogicQA, HotpotQA | A* wins on StrategyQA (+10.3 pts), HotpotQA (+3.3 pts); loses on LogicQA (−0.9 pts) | Partial (LogicQA loss) | Compact-logic tasks punish branching — task topology, not algorithm fault |
| 2 | **Oracle Gap Analysis** | Measure upper bound of strategy selection | Instance-level decomposition (Both/CoT-only/A*-only/Neither) | Oracle is +9.79 (Strateg.), +13.51 (Logic), +14.10 (Hotpot) above best single method | Not a failure — confirms routing is the bottleneck | — |
| 3 | **Topology Clustering** | Cluster problems by structural features | K-means (k=4) on 13 hand-crafted features, 744 paired items | 4 clusters found (C1: linear, C2: mid-branch, C3: complex, C4: deep-branch) | **Partial** | Clusters correlate with topology but do NOT cleanly separate A*-wins from CoT-wins — both C3 and C4 are "Both wrong" dominated; clustering alone is not a ready router |
| 4 | **Feature Importance** | Identify which problem features predict winner | Random Forest on binary (A*-wins vs CoT-wins) | Top features: ctx_len (0.208), q_len (0.194), ctx_sents (0.113) | **Partial** | Features are symbolic/heuristic — no model access. Signal is real but weak |
| 5 | **Topology Classifier** (Router) | Route to A* or CoT using only problem structure | Logistic Regression, RF, GB, DT — 5-fold CV | Best: LogReg at 51.1% ± 3.2% CV; recovers **26.1%** oracle gap on StrategyQA | **Near-failure** | 51.1% is barely above coin-flip (50%). Features don't capture instance-level model compatibility |
| 6 | **Fluency–Correctness Asymmetry** | Quantify if trace fluency misleads routing | Heuristic fluency proxies on 1,300 CoT+A* trace pairs | CoT more fluent in **38.0%** of divergent cases; A* globally more fluent (0.725 vs 0.673) | Measurement is proxy-based | No ground-truth fluency labels; human validation needed to confirm the 38% figure is causal |
| 7 | **SLM Amplification** | Does A* help small models more? | Qwen-0.5B on all 3 benchmarks | HotpotQA: +21.6 pts (+257%); StrategyQA: ≈0; LogicQA: −3.8 pts | **LogicQA failed** | Compact-logic topology degrades Qwen-0.5B even more than large model — topology drives the failure |
| 8 | **Router Failure Analysis** | Characterise A*-failure cases | Manual inspection of 45 A*-fail-CoT-win cases | 91.1% = overthinking simple problems; 8.9% = premature commitment | Analysis limited | Only 45 cases inspected; generalisation to other domains/benchmarks needs validation |
| 9 | **Efficiency Analysis** | Is A*'s token cost justified? | Estimated tokens from avg node counts; Acc/1K tokens | A* uses ~4.9× more tokens; beats self-consistency accuracy at same budget | Not a failure — expected cost | Efficiency claim depends on token estimate (60 tok/step); not measured directly |

---

## Slide 4 — What Failed and Why (Critical)

### The Three Real Failures

#### ❌ Failure 1: Topology Classifier Too Weak (51.1% accuracy)
- **What we expected**: Structural features (branching, depth, linearity) would strongly predict which strategy wins
- **What happened**: Best CV accuracy is 51.1% — essentially coin-flip level
- **Why**: Topology features are *necessary but not sufficient*. Two problems can have identical topology (same branching complexity, same context length) but one aligns with the model's parametric knowledge and the other does not. The features capture structure but **not the model–problem compatibility that actually determines the winner**
- **Implication**: Pre-generation routing needs learned embeddings, not symbolic features

#### ❌ Failure 2: A* Degrades LogicQA at Both Model Sizes
- **What we expected**: A* would be neutral or slightly helpful on compact multiple-choice logic
- **What happened**: LLaMA-3.1-8B loses 0.9 pts; Qwen-0.5B loses 3.8 pts
- **Why**: LogicQA requires maintaining a globally coherent logical frame across all steps. A*'s branching **fragments this frame** — each branch independently revisits assumptions, causing the argument structure to collapse. This is not a capability failure; it is a topology mismatch.
- **Implication**: Any system deploying A*-style search should explicitly gate on compact-logic detection

#### ❌ Failure 3: Fluency Measurement is Proxy-Only
- **What we expected**: Quantify the exact rate at which surface fluency misleads selecting the worse trace
- **What happened**: We measured 38.0% asymmetry using heuristic sentence-length and vocabulary-diversity proxies
- **Why it's incomplete**: These proxies cannot distinguish fluency from coherence. True measurement requires human annotation or a calibrated fluency model. The 38% is directionally correct but not publication-reliable without that validation.
- **Implication**: Needs one round of human annotation on ~200 divergent cases to make this result airtight

### One Root Cause Beneath All Three
> **We can predict problem topology, but we cannot yet predict model–problem compatibility.** That gap — 73.9% of the oracle gap remaining — is the open problem this work defines.

---

## Slide 5 — Insights and Next Steps

### What Worked and Why

| Insight | Evidence | Mechanism |
|---|---|---|
| A* dramatically amplifies SLMs on multi-hop (HotpotQA +257%) | Qwen-0.5B: 8.4% → 30.0% | SLM lacks parametric integration; search substitutes working memory by keeping partial hypotheses alive |
| 91% of A*-failures are overthinking | 41/45 manually annotated cases | Search explores sub-structure where none is needed; the sufficiency signal is not detected |
| Topology features predict direction of benefit | ctx_len, q_len are top-2 features | Long context + many sentences = distributed evidence = search wins; short, self-contained Q = CoT wins |
| Pre-generation routing beats post-generation routing | Topology router 26.1% oracle recovery vs RF 14.8% | Knowing structure before generating traces is more efficient than comparing generated traces |

### What Didn't Work and What It Tells Us

| Failure | Implication | Next Design Choice |
|---|---|---|
| Symbolic topology features too weak as classifier | Problem structure alone ≠ routing decision | Learn topology embeddings from question encoders (e.g., BERT sentence rep → router) |
| A* hurts compact-logic (LogicQA) | Global coherence incompatible with branching | Add topology gate: detect MCQ + short-context → skip search entirely |
| Fluency proxies not ground-truth | Fluency–Correctness Asymmetry claim is directional, not definitive | 200-example human annotation study; or fine-tune a fluency classifier on annotated pairs |
| 73.9% oracle gap still unreached | Routing is fundamentally unsolved at instance level | Model–problem compatibility score: measure how well model parameters support the retrieval operations the problem demands |

### The Open Problem (What We've Defined)
> Build a router that knows not just the structure of the problem, but whether *this model* can handle *this structure* — before generating a single token.

That is the research frontier this paper establishes.

---

## Slide 6 — Last Week: RAG Experiments

### What We Built
A FAISS-based Retrieval-Augmented Reasoning (RAR) pipeline across all 3 benchmarks.

**Idea**: Instead of generating reasoning from scratch, retrieve the correct reasoning trace from a similar past question and use it as an in-context example.

**Setup**: Knowledge base = correct CoT + A* traces from prior runs. Embeddings via `sentence-transformers/all-MiniLM-L6-v2`. Model: LLaMA-3.1-8B-Instruct. Evaluated on 200 questions per dataset.

### Results vs Baselines

| Dataset | CoT | A* | **RAG (best genuine)** | RAG inflated* |
|---|---|---|---|---|
| HotpotQA  | 76.8% | 80.1% | **21.0%** | 40.0%* |
| StrategyQA | 64.5% | 74.8% | **71.0%** | — |
| LogicQA   | 45.8% | 44.9% | KB built, eval not completed | — |

*\*40.0% used `allowSelf=True` — retrieved the same question being evaluated. This is test leakage, not a real result.*

### Variants Tested on HotpotQA (13 configs)
| Config | Accuracy | Finding |
|---|---|---|
| Baseline (top-1, exclude self) | 20.0% | Starting point |
| F1 filter ≥0.30 | 18.5% | Stricter KB hurts (fewer retrievals) |
| F1 filter ≥0.40 | 18.0% | Worse still |
| F1 filter ≥0.50 | 21.0% | Best genuine: quality > quantity |
| top-3, hybrid alpha=0.2 | 20.5% | Marginal gain from multiple retrievals |
| top-5 | 20.0% | No gain from more retrievals |
| **allowSelf=True** | **40.0%** | **Inflated — confirms retrieval is the bottleneck** |

### ❌ Why HotpotQA RAG Failed (20% vs 80.1% A*)

**Root cause: Retrieval retrieves structure, not facts**

HotpotQA requires evidence-specific multi-hop chaining (e.g., "actress who played X → government role she held"). Two questions that *read* similarly (both are multi-hop entity chains) have completely different factual paths. The retrieved reasoning trace is about the wrong entities — it doesn't help; it misleads.

| Failure Reason | Evidence |
|---|---|
| KB coverage too small | 147 KB entries for 500 questions (29% coverage) |
| FAISS embeds question text only | Semantically similar ≠ factually overlapping for multi-hop |
| Retrieved traces about different facts | Confuses generation rather than guiding it |
| **allowSelf jump to 40%** confirms it | When model retrieves its own correct trace, it works — meaning *retrieval quality is the entire problem* |

### ✅ Why StrategyQA RAG Worked (71%, near A*'s 74.8%)
- Commonsense patterns **generalise** across similar questions ("Does X have Y?" shares reasoning structure regardless of specific X, Y)
- Yes/no format is format-compatible across retrieved examples
- 127-entry KB covers 110 unique questions — reasonable density

### Key Takeaway
> RAG works when retrieved traces are structurally and factually transferable. It fails when answer generation requires question-specific facts not present in the retrieved trace. For HotpotQA, the fix is not retrieval tuning — it is **evidence-grounded retrieval**: index evidence passages alongside questions, not question text alone.
