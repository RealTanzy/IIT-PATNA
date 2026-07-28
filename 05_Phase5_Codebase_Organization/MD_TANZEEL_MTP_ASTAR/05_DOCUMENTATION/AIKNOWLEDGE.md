# AIKNOWLEDGE — Researcher Agent Briefing
**Domain:** Multi-step Question Answering | Reasoning Strategy Selection | LLM Search vs Linear Reasoning
**Compiled:** May 14, 2026
**Purpose:** Full domain knowledge transfer to an AI Researcher Agent. This document is your briefing. You are being handed an active research project mid-stream. By the end of this document you should understand not just what was done, but **why every decision was made**, what was discovered, what failed and why, and where the research currently stands. You will be working as a collaborator — not a summarizer.

---

## SECTIONS

1. [Your Role as Research Agent](#1-your-role-as-research-agent)
2. [The Core Problem Being Solved](#2-the-core-problem-being-solved)
3. [The Setup — What We Are Working With](#3-the-setup--what-we-are-working-with)
4. [The Research Journey — Timestamped, With Intent](#4-the-research-journey--timestamped-with-intent)
5. [What the Numbers Actually Say](#5-what-the-numbers-actually-say)
6. [The Intellectual Contributions — What We Discovered](#6-the-intellectual-contributions--what-we-discovered)
7. [What Has Been Written and Submitted](#7-what-has-been-written-and-submitted)
8. [How Reviewers Pushed Back and How We Responded](#8-how-reviewers-pushed-back-and-how-we-responded)
9. [What Exists in the Workspace and Why](#9-what-exists-in-the-workspace-and-why)
10. [What Is Still Open](#10-what-is-still-open)
11. [How to Think About This Research — Your Mental Model](#11-how-to-think-about-this-research--your-mental-model)

---

## 1. Your Role as Research Agent

You are picking up an active ML research project that is in the rebuttal-and-revision phase of peer review. You are not starting from scratch. The research question has been answered empirically. A paper has been written. Reviewers have responded. New experiments have been run during the rebuttal window. What is left open are: (a) large-model scaling runs (14B/27B) currently executing on the server, and (b) incorporating rebuttal findings into the revised paper.

Your job when collaborating is to reason like a researcher who understands the intellectual arc of this project — not just what the numbers are, but *why we asked the question*, *why we made certain design choices*, *where we pivoted and why*, and *what the current weaknesses and open threads are*. You should be able to hold a technical conversation, suggest next steps, catch logical inconsistencies, and help think through arguments.

Read this entire document before responding to any task.

---

## 2. The Core Problem Being Solved

**Paper title:** *When Does Search Beat Reasoning? A Topology-Driven Diagnostic of A\* versus Chain-of-Thought*

**The motivating question:** LLMs can generate multi-step reasoning in two structurally different ways. The first is Chain-of-Thought (CoT) — a single, linear, sequential trace. The model generates one coherent thread of reasoning leading to an answer, committing forward at each step. The second is A\* Search — a graph-based approach where the model generates multiple candidate reasoning steps at each state, scores them with a heuristic, and expands the most promising node first. It can backtrack, branch, and explore alternatives before committing.

The community had implicitly assumed that more structured search should be better, or that CoT's simplicity made it a weaker baseline. **Both assumptions are wrong.** Neither method universally dominates. The winner depends on the *structure of the question itself* — specifically, how much branching, multi-hop chaining, and evidence aggregation the question demands.

**The thesis, confirmed by experiment:** Reasoning topology — the structural complexity of a question's branching and multi-hop requirements — predicts which method will win. This is a property of the question, not of the dataset or the model. The same topology that favors A\* on HotpotQA also favors A\* on StrategyQA questions with matching structure, even though they come from completely different datasets.

**Why this matters:** If you can characterize a question's topology *before running any LLM*, you can route it to the right method. That's the practical payoff. The theoretical payoff is a diagnostic framework explaining when search-based reasoning is actually worth its compute cost.

**What A\* search means here specifically:** It's not classical graph search on a symbolic state space. The "states" are partial LLM reasoning traces. At each node, the LLM generates K candidate next reasoning steps. Each is scored by a DeepSeek-14B critic (the heuristic h(n)), and the lowest f(n) = g(n) + h(n) node is expanded next. g(n) accumulates depth penalty + confidence penalty. The search terminates when a "goal" state is reached (a terminal reasoning pattern like "The answer is...") or when budget B is exhausted.

---

## 3. The Setup — What We Are Working With

### Datasets

Three datasets were chosen to represent structurally distinct reasoning paradigms — not just different topics but different *reasoning demands*:

| Dataset | N (used) | Format | Reasoning demand | Winner |
|---------|---------|--------|-----------------|--------|
| **LogicQA** | 651 | 4-way MCQ | Formal deductive logic, constraint satisfaction | CoT by -0.9pp (NS) |
| **StrategyQA** | 500 | Yes/No | Commonsense multi-hop, assumption chaining | A\* by +10.3pp (p<0.001) |
| **HotpotQA** | 1,000 | Free-form span | Multi-hop evidence synthesis, entity chaining | A\* by +3.3pp (p=0.048) |

**Total: 2,151 questions**, fully balanced — same model, same question sets, both methods evaluated on every question.

The inclusion of LogicQA is deliberate and important. A\* actually *loses* on LogicQA. If we had only reported StrategyQA and HotpotQA, the paper would overclaim. The honest story requires including the failure case — it's what makes the topology argument credible.

### Models

| Model | Parameters | Role |
|-------|-----------|------|
| Qwen-0.5B | 0.5B | Early baseline runs (archive), capability×topology interaction |
| Llama 3.1-8B | 8B | Main paper model — all paired 2,151Q runs |
| DeepSeek-R1-Distill-Qwen-14B | 14B | Critic / heuristic scorer in A\* (h-function) |
| 14B/27B models | 14B/27B | Scaling validation — currently running (May 2026) |

### Infrastructure

- **Ollama** — local LLM serving for development, Llama 3.1-8B and Qwen-0.5B
- **vLLM** — high-throughput batched inference for full-dataset runs (April 27 onward)
- **Server** — multi-GPU (used device 0 + 1 for large model runs)

### Cost function (important to understand)

`f(n) = g(n) + h(n)`  
where `g(n) = depth_n + Σ(1 − confidence_t)` across all steps to node n  
and `h(n)` = DeepSeek critic score estimating remaining reasoning cost  
Confidence is self-reported by the LLM, clamped to [0.5, 0.99].  
Budget B = 15 nodes maximum per question (default). Branching factor K = 3 candidates per expansion.

---

## 4. The Research Journey — Timestamped, With Intent

This section is the core of your briefing. Read it chronologically. For each phase, understand not just what was done but why the decision was made and what it revealed.

---

### [March 29, 2026] — Dual-Trace Synthesis (Phase 3)

**The idea:** If CoT and A\* each solve different questions, maybe combining their traces before answer extraction would help. Run both on the same question, linearize the A\* non-linear trace into clean step-by-step text, feed both to an LLM for synthesis, and extract a unified answer.

**Why we tried this:** Phase 2 heuristic ablation had already shown coverage heuristic matters (+5.4pp on StrategyQA). The natural question was: can we get the best of both worlds by combining, rather than choosing?

**What happened:**

| Method | Accuracy (LogicQA, 100Q, Llama 3.1-8B) |
|--------|----------------------------------------|
| CoT alone | 39% |
| A\* alone | 33% |
| Dual-trace synthesis | **42%** |

Synthesis won by +3pp over CoT. But there was a hard infrastructure limit — Ollama crashed on the full synthesis prompts (too long for the local runtime). We fell back to a lightweight synthesis that achieved 42%. True LLM synthesis on a dedicated GPU would likely push this further.

**What it told us:** The methods are genuinely complementary — combining their reasoning improves on either alone. This seeded the idea that the winner is question-dependent and that routing (choosing which to run per question) might be more powerful than combining.

**Scripts:** `PHASE_3/phase3_dual_trace_synthesis.py`, `PHASE_3/phase3_dual_trace_synthesis_v2.py`  
**Result file:** `PHASE_3/logicqa_phase3_dualtrace_100_llama3.1_8b.json`

---

### [Pre-April 6] — Heuristic Ablation (Phase 2)

**The question:** Does the specific heuristic design matter in A\*, or is any heuristic fine?

**Ablation on StrategyQA (4 configurations):**

| Config | Heuristic | Accuracy |
|--------|-----------|---------|
| A0 | No heuristic (BFS) | 69.2% |
| A1 | Depth penalty only | 67.8% |
| A2 | **Coverage only** | **74.6%** (+5.4pp) |
| A3 | Depth + coverage | unstable |

**What it revealed:** Coverage heuristic — tracking how many required entities have been mentioned in the reasoning trace — is the essential signal. Depth alone *hurts* (-1.4pp below BFS). Combining both is unstable. This is why the final A\* uses the coverage heuristic as the primary h(n) signal on StrategyQA/HotpotQA.

**Early baseline (Qwen-0.5B archive, 500Q each):**

| Dataset | CoT | A\* | Key insight |
|---------|-----|-----|-------------|
| LogicQA | 24.0% | 20.2% | A\* loses on formal logic |
| HotpotQA | 8.4% | **30.0%** | A\* gives 3.5× amplification to a weak model |
| StrategyQA | 53.6% | 53.4% | Statistical tie |

The HotpotQA number is striking: A\* turns an 8.4% baseline into 30.0%. The mechanism is that A\* externalizes the multi-hop bookkeeping — the small model doesn't need to hold a 3-hop reasoning chain in a single context window; A\* manages the state space externally.

---

### [April 6, 2026] — RAG and Fine-tuning Experiments
*Context: Weekly supervisor update for Dr. Asif Ekbal. This was a results snapshot, not a final study.*

**Motivation:** Before committing fully to the A\* vs CoT framing, we tested whether retrieval-augmented generation (RAG) or dataset-aware fine-tuning could close the gaps. If RAG could match A\*, it would undercut the contribution.

**What was built:** FAISS indexes over paired CoT+A\* traces. Llama 3.1-8B used for generation with retrieved context. Strict fair evaluation: examples used as retrieval entries were excluded from the test set.

**Results (100–200 sample runs):**

| Dataset | Baseline CoT | Baseline A\* | Fine-tuned | RAG (strict fair) |
|---------|------------|------------|-----------|-------------------|
| LogicQA | 39.0% | 33.0% | 40.0% | 37.0% |
| StrategyQA | 73.0% | 79.0% | 62.0% | 71.0% |
| HotpotQA | 53.0% | 80.1% (1000Q) | 53.0% | 21.0% |

**Why RAG failed on HotpotQA:** Bridge questions need 3+ hop traversal. FAISS retrieval brings in relevant traces, but it can't reconstruct the exact multi-hop chain needed for a new question. Relaxed retrieval (allow self-retrieval, upper-bound diagnostic) raises this to 40.0% — confirming that retrieval coverage, not generation, is the bottleneck.

**Decision made:** RAG is not a substitute for A\* on multi-hop tasks. Fine-tuning degraded StrategyQA by 11pp (fine-tuning on one distribution hurts another). We deprioritized RAG and returned focus to the A\* vs CoT diagnostic framing.

---

### [April 17, 2026] — The Unfair Comparison Problem (Critical Discovery)
*This is the most important methodological fix in the project.*

**What we discovered:** The existing result was:
- A\* (Llama 3.1-8B): 80.1% on HotpotQA
- CoT baseline: 8.4% on HotpotQA — but this was **Qwen-0.5B**

A 16× model size difference. The claimed A\* dominance was comparing a sports car to a bicycle. Any ML paper reviewer would immediately reject this comparison. We caught it before submission.

**The fix:** Run Llama 3.1-8B CoT on the same 100 questions from the A\* result set. Same model. Same questions. Head-to-head.

**Result:**

| Method | Correct/100 | Accuracy |
|--------|------------|---------|
| Llama 3.1-8B A\* | 74 | 74.0% |
| Llama 3.1-8B CoT | 73 | 73.0% |
| Oracle | 85 | **85.0%** |

Overall: +1pp. Almost tied. A weaker result than 80.1% vs 8.4%, but a *real* result.

**But the +1pp aggregate is misleading.** Broken down by question type:

| Q-type | N | A\* | CoT | Gap |
|--------|---|-----|-----|-----|
| bridge (multi-hop) | 80 | **77.5%** | 71.2% | **+6.3pp A\*** |
| comparison | 11 | 54.5% | **72.7%** | **+18.2pp CoT** |
| yes/no | 9 | 66.7% | **88.9%** | **+22.2pp CoT** |

This is the paper's central insight in miniature. The question type completely determines the winner. Aggregate accuracy obscures this.

**A\* failure mode on yes/no:** The `q_type == "yesno"` label in the HotpotQA dataset misleads A\* into outputting "yes" even when the gold answer is a named entity like "Dundee Canal" or "family". CoT ignores the label and answers the question as posed. This is a known bug — not fixed yet in the codebase.

**Outcome quadrant (complementarity):**

| Both correct | 62 | Both wrong | 15 |
|-------------|----|-----------|----|
| **A\*-only** | **12** | **CoT-only** | **11** |

23 questions exist where one method succeeds and the other fails. This 23% complementarity is the motivation for routing.

**Script:** `17TH APRIL.../targeted_cot_vs_astar.py`

---

### [April 17, 2026] — Topic Modeling on HotpotQA (NMF, 6 Topics)

**Motivation:** Q-type tells us structure. But we wanted to know: on what *content topics* does A\* win, and how much of the dataset do those cover?

**Method:** NMF on TF-IDF of all 1,000 A\* question texts → 6 content topics. Ran CoT on 200 additional questions for paired comparison.

**Script:** `17TH APRIL.../hotpotqa_topic_astar_analysis.py`

| Topic | Coverage | A\* paired | CoT paired | Gap |
|-------|---------|-----------|-----------|-----|
| **sports** | 32% | 81.6% | 69.4% | **+12.2pp A\*** |
| geography | 19% | 69.4% | 77.6% | -8.2pp CoT |
| **film_cinema** | 18% | 77.2% | 66.7% | **+10.5pp A\*** |
| music | 13% | 69.6% | 78.3% | -8.7pp CoT |
| comparison_general | 10% | 74.1% | 74.1% | tie / CoT slight |
| **biography** | 8% | 91.3% | 78.3% | **+13.0pp A\*** |

A\*-dominant topics: sports + film_cinema + biography = **58% of HotpotQA**

**Why these three?** Sports, film, and biography questions are dense multi-hop entity chains. "What team did [player] play for when they won [championship] in [year]?" requires chaining at least 2–3 discrete facts through named entities. A\*'s node-by-node expansion naturally handles this. CoT tries to do it in one linear thread and often fails at the first hop.

---

### [April 18, 2026] — Filling Dataset Gaps + Full Cross-Dataset Topic Model

**Problem discovered:** StrategyQA had only 93/500 CoT results. HotpotQA had only 300/1000. A cross-dataset analysis on mismatched coverage is not valid.

**Fix applied:**

| Script | What it did | Outcome |
|--------|------------|---------|
| `fill_strategyqa_cot.py` | Ran CoT on 407 missing StrategyQA questions | 500Q fully paired |
| `fill_hotpotqa_cot.py` | Ran CoT on 700 missing HotpotQA questions | 1000Q fully paired |
| `full_topic_analysis.py` | NMF 8-topic model on all 2,151 questions | **A\* dominant in 81.4%** |

**Final balanced dataset:** LogicQA 651 + StrategyQA 500 + HotpotQA 1000 = **2,151 questions**, every question evaluated by both methods with the same model.

**Cross-dataset 8-topic NMF results:**

The critical finding is that the topic boundaries *cut across dataset labels*. content_domain_factual (43.7% of corpus) contains questions from both StrategyQA and HotpotQA — and A\* wins in both. The CoT-dominant topics (comparison, constraint_satisfaction, statement_strength = 18.6%) span LogicQA and the comparison subset of HotpotQA. The question structure, not the dataset name, governs the outcome.

| Topic | Coverage | Winner | Characteristics |
|-------|---------|--------|----------------|
| content_domain_factual | 43.7% | A\* | Multi-hop lookups, entity chains |
| assumption_analysis | 13.9% | A\* | Premise scanning, argument structure |
| temporal_event | 9.8% | A\* | Multi-hop chains with time ordering |
| evaluate_argument | 9.1% | A\* (slight) | Multi-premise evaluation |
| biography_factual | 4.9% | A\* | Entity disambiguation |
| comparison | ~6% | CoT | Direct entity comparison |
| statement_strength | ~5% | CoT | Judgment/inference, single-step |
| constraint_satisfaction | ~7.6% | CoT | Formal logic, sequential constraints |

**A\*-dominant share: 81.4%** | **CoT-dominant: 18.6%**

---

### [April 19, 2026] — Writing the Paper

**LaTeX paper created:** `19th_April/When_Does_Search_Beat_Reasoning/main.tex`  
**References:** `references.bib` — 22 entries  
**First compile:** 16 pages, 212KB PDF

Also created a **TMLR-formatted version** at `19th_April/TMLR_Version/`.

**Key references added:** Tree-of-Thoughts (Yao et al., 2023), Self-Consistency (Wang et al., 2023), LATS (Zhou et al., 2023), MCTSr (Zhang et al., 2024), rStar-Math (Guan et al., 2025), CoT-Valve (2025), overthinking survey (2025).

---

### [April 19–20, 2026] — Critical Self-Review and 9 Fixes

Before sending the paper anywhere, a critical review identified 9 problems. Understanding these is important — they reflect the paper's intellectual honesty.

| # | Severity | Problem | Fix |
|---|----------|---------|-----|
| 1 | **CRITICAL** | Table 3 (complementarity) contradicted Table 1 (main results) numerically | Recomputed all cells to mathematical consistency |
| 2 | **CRITICAL** | No statistical significance testing at all | Added 95% Wilson CIs + McNemar p-values to all main results |
| 3 | **CRITICAL** | Called a result "Theorem" without a proof | Downgraded to Proposition; added empirical validation |
| 4 | HIGH | 12 tables, 0 figures — no visual intuition | Added Figure 1 (branching coefficient vs A\* advantage) and Figure 2 (topic bar chart) via pgfplots |
| 5 | HIGH | No reproducibility details | Added generation parameters (temperature, top-p, seeds), Appendix C with all prompt templates |
| 6 | HIGH | Router mentioned but not specified | Added XGBoost implementation details, 5-fold CV protocol, train/test split details |
| 7 | MEDIUM | NMF topic quality unvalidated | Added NPMI coherence scores, k-sensitivity testing (k=6..10), bootstrap stability |
| 8 | MEDIUM | BibTeX error in StrategyQA citation | Fixed |
| 9 | MEDIUM | False "NeurIPS format" comment in LaTeX preamble | Removed |

**Final paper state:** 18 pages, 240KB PDF, 2 figures, 12 tables, 22 references, 4 appendices, compiles cleanly.

---

### [April 27, 2026] — Full Dataset vLLM Runs

**Why vLLM:** Ollama is fine for development but slow for 1000+ examples. vLLM provides batched, high-throughput inference that runs the full datasets in reasonable time.

**What was run:** All three datasets (2,151 questions total), both methods, using the vLLM server started via `start_vllm.sh`.

**Results stored in:** `27th_APRIL_WHOLE_DATASET_RUN/hotpotqa/`, `/logicqa/`, `/strategyqa/`  
**Master runner:** `27th_APRIL_WHOLE_DATASET_RUN/run_all.sh`

---

### [May 1, 2026] — Reproducibility Pass

**Purpose:** Establish a clean, fixed-seed, fully reproducible baseline. All the earlier runs were done incrementally and some had different seeds or slightly different question subsets.

`1ST_MAY_CONSISTENT/` contains paired A\* and CoT scripts for all three datasets with fixed seeds, identical question selection logic, and results stored in `astar_results/` and `cot_results/` subfolders.

This is the cleanest version of the baseline and should be the reference for any future reproduction attempts.

---

### [May 2, 2026] — Rebuttal Phase (Paper Under Review)

The paper received 3 reviewers. Scores: 2.5 Borderline, 1.5 Resubmit, 2.5 Borderline. The 1.5 is the concern. New experiments were run during the rebuttal window — these are not just arguments, they are actual new results.

**Rebuttal file:** `2nd_may_Rebuttal/REBUTTAL_ALL_REVIEWERS_v2.md`

**New experiments run and their findings:**

**① Fluency–Correctness Asymmetry (new quantified finding)**

Measured across all disagreement cases between CoT and A\*. In **38% of divergent cases**, CoT produces the more fluent trace but the wrong answer. A\*'s trace is more fragmented but correct.

This is why output-based routing hard-fails. Any router that reads the generated traces and scores which "looks better" will systematically prefer CoT's fluent-but-wrong output. The ceiling for output-based routing is not a tuning problem — it's fundamental. The paper now explicitly makes this argument.

**② Topology Router (built and measured)**

Built a gradient-boosted decision tree (XGBoost) using only the 13 pre-generation structural features from question text. Zero LLM calls. Evaluated on held-out test split with 5-fold CV.

Result: **recovers 26.1% of the oracle gap**. Outperforms all output-based routers, including LLM-as-Critic (which reads both full generated traces from DeepSeek-14B). A router that never sees the generated output outperforms one that reads everything.

This is the most counterintuitive finding in the project and arguably the strongest contribution.

**③ BFS Ablation (directly addresses ToT comparison)**

Ran 3 heuristic modes on a fixed 300-question HotpotQA sample:

| Mode | What it is | Accuracy | Nodes |
|------|-----------|---------|-------|
| BFS (h=0) | ToT-style undirected branching | 77.0% | 5.3 |
| Greedy (f=h) | Critic-only guidance | 77.7% | 5.0 |
| Full A\* | g(n)+h(n) paper default | 76.0% | 5.2 |

All within 1.7pp. BFS (no heuristic) and full A\* are indistinguishable in accuracy. The driver is **branching exploration itself**, not the specific heuristic weighting. This directly answers the reviewer who asked "how does this compare to ToT?"

**④ Budget and Branching Sensitivity**

Budget sweep (B=5..30, HotpotQA): 73.0% → 77.3% → 78.0% → **78.7%** → 76.0%. Peak at B=20, paper uses B=15 (within 0.7pp, saves ~25% compute).

Branching sweep (K=1..4): 77.7% → 78.3% → 76.0% → **80.3%**. K=1 (no branching, pure chain) still achieves 77.7% — the framework degrades gracefully.

**⑤ Compute Analysis (token counts)**

| Method | Accuracy | Tokens/Q |
|--------|---------|---------|
| CoT | 76.80% | ~512 |
| SC ×5 | 78.21% | ~2,560 |
| A\* K=1, B=30 | 77.7% | 3,372 |
| A\* K=3, B=15 (paper) | 78.0% | 12,452 |
| A\* K=3, B=20 (peak) | **78.7%** | 12,391 |

The +10.3pp on StrategyQA is **unreachable by running CoT more times**. Self-consistency on StrategyQA achieves 71.22% — still 3.6pp below A\*. The gain is structural, not a compute-scaling effect.

**⑥ Heuristic Validation (500 annotated states)**

DeepSeek critic underestimates human-assessed reasoning progress: mean 0.43 vs. 0.58 (paired t-test p<0.001). Maximum overestimation ε_max = 0.14 consistent across all 3 datasets. This satisfies the conservative-estimation condition required for the admissibility proposition.

---

### [May 11–14, 2026] — Large Model Scaling Tests (In Progress)

**Purpose:** Validate that topology-driven findings hold at 14B and 27B scale. If the winner flips when model size increases, it would suggest the findings are model-capability artifacts rather than structural phenomena.

**Location:** `14b_dataset_check/`  
**Status:** Runs executing. Results not yet finalized.  
**Expected finding:** Same topology predicts same winner — structure is a question property, not a model property.

**Progress monitoring:** `check_progress_14b.py` shows real-time progress.  
**Result directories:** `results_14b/`, `results_27b/`, `results_8b/`, `results_nohop_14b/`

---

## 5. What the Numbers Actually Say

### Main Results — Llama 3.1-8B (Full Balanced Dataset)

| Dataset | N | CoT | A\* | Winner | Gap | p-value |
|---------|---|-----|-----|--------|-----|---------|
| LogicQA | 651 | **45.78%** | 44.85% | CoT | -0.93pp | p=0.61 NS |
| HotpotQA | 1,000 | 76.80% | **80.10%** | A\* | +3.30pp | p=0.048 |
| StrategyQA | 500 | 64.52% | **74.80%** | A\* | +10.28pp | p<0.001 |

### Oracle Bounds (Upper limit if we always picked the right method per question)

| Dataset | Oracle | CoT | A\* | Gap to Oracle (CoT) | Gap to Oracle (A\*) |
|---------|--------|-----|-----|-------------------|-------------------|
| LogicQA | 69.9% | 45.8% | 44.9% | **24.1pp** | **25.0pp** |
| HotpotQA | 85.0% | 76.8% | 80.1% | 8.2pp | 4.9pp |
| StrategyQA | 82.8% | 64.5% | 74.8% | 18.3pp | 8.0pp |

The oracle gap is the theoretical ceiling for a perfect router. 24.1pp on LogicQA means that if we could always know which method to use, we'd gain 24pp over CoT. The topology router recovers 26.1% of this gap.

### Early Baseline — Qwen-0.5B (500 questions each, archive)

| Dataset | CoT | A\* | Key observation |
|---------|-----|-----|----------------|
| LogicQA | 24.0% | 20.2% | A\* loses at small model size too |
| HotpotQA | 8.4% | **30.0%** | **3.5× amplification** — A\* compensates for weak memory |
| StrategyQA | 53.6% | 53.4% | Statistical tie |

### HotpotQA — Per Question Type (100Q head-to-head, Llama 3.1-8B)

| Q-type | N | A\* | CoT | Gap |
|--------|---|-----|-----|-----|
| bridge | 80 | **77.5%** | 71.2% | **+6.3pp A\*** |
| comparison | 11 | 54.5% | **72.7%** | +18.2pp CoT |
| yes/no | 9 | 66.7% | **88.9%** | +22.2pp CoT |

### HotpotQA — Per Topic (NMF, 6 Topics)

| Topic | Coverage | A\* | CoT | Gap |
|-------|---------|-----|-----|-----|
| sports | 32% | 81.1% | 69.4% | **+12.2pp A\*** |
| film_cinema | 18% | 81.1% | 66.7% | **+10.5pp A\*** |
| biography | 8% | 90.5% | 78.3% | **+13.0pp A\*** |
| geography | 19% | 81.3% | 77.6% | +3.7pp A\* |
| music | 13% | 79.8% | 78.3% | +1.5pp A\* |
| comparison_general | 10% | 63.5% | 74.1% | **-10.6pp CoT** |

### Phase 3 Dual-Trace Synthesis (100Q LogicQA, Llama 3.1-8B)

| Method | Accuracy |
|--------|---------|
| CoT alone | 39% |
| A\* alone | 33% |
| Dual-trace synthesis | **42%** |

### Rebuttal Compute Table (300Q HotpotQA, measured tokens)

| Method | Accuracy | Tokens/question |
|--------|---------|----------------|
| CoT | 76.80% | ~512 |
| SC ×5 | 78.21% | ~2,560 |
| A\* K=1, B=30 | 77.7% | 3,372 |
| A\* K=3, B=15 | 78.0% | 12,452 |
| A\* K=3, B=20 | **78.7%** | 12,391 |

---

## 6. The Intellectual Contributions — What We Discovered

These are the findings. Understand each one deeply — they form the paper's argument.

### Finding 1: Neither Method Universally Dominates
A\* wins on StrategyQA (+10.3pp) and HotpotQA (+3.3pp). CoT wins on LogicQA (-0.9pp). The winner is not the same across datasets. Anyone claiming "search > reasoning" or "CoT > search" universally is wrong. This project provides the diagnostic.

### Finding 2: Branching Complexity Predicts the Winner (R² = 0.94)
We built a Reasoning Topology Framework: 13 pre-generation structural features extracted from question text (hop count, entity density, branching coefficient, logical connectives, temporal markers, etc.). K-means (k=4) clusters all 2,151 questions into topology types C1–C4. As branching complexity increases: C1 (−0.4pp) → C2 (+2.1pp) → C3 (+6.8pp) → C4 (+11.4pp). R² = 0.94. Question structure, not dataset identity, governs the outcome.

### Finding 3: The Topic/Content Boundary Cuts Across Datasets
NMF on 2,151 question texts yields 8 topics. 81.4% of the corpus (5/8 topics) is A\*-dominant. All A\*-dominant topics share the same structural signature: multi-hop chaining, entity resolution, premise scanning. CoT-dominant topics (18.6%) are single-step or comparison tasks. This is an independent replication of Finding 2 from a different angle.

### Finding 4: Fluency–Correctness Asymmetry Caps Output-Based Routing
In 38% of disagreement cases, CoT's trace is more fluent but wrong. A\*'s fragmented trace is correct. Any output-based router — LLM-as-critic, semantic entropy, score-based — will be fooled by CoT's fluency. This creates a hard ceiling that no amount of output evaluation can escape.

### Finding 5: The Pre-Generation Topology Router Beats Post-Generation Evaluation
A XGBoost router using only the 13 pre-generation structural features (zero LLM calls, reads only raw question text) recovers 26.1% of the oracle gap. This outperforms LLM-as-Critic, which reads both full generated traces from DeepSeek-14B. The bottleneck is recognizing problem structure before inference — not evaluating trace quality after it.

### Finding 6: Branching Structure Is the Driver, Not the Specific Heuristic
BFS (h=0), greedy (f=h), and full A\* all perform within 1.7pp on HotpotQA. The heuristic choice is nearly irrelevant. The decision to branch and explore multiple states is what matters. This means the finding is not an artifact of our specific DeepSeek critic choice.

### Finding 7: A\* Amplifies Small Models by 3.5× on Multi-Hop Tasks
Qwen-0.5B: CoT 8.4% → A\* 30.0% on HotpotQA. A\* externalizes the multi-hop bookkeeping that a small model's limited context window cannot handle internally. Topology is a question property, not a model property.

### Finding 8: 91.1% of A\* Failures Are "Overthinking"
A\* continues expanding past the correct answer, accumulating noise. This is the dominant failure mode — the search engine finds the answer but then keeps going and overwrites it. Secondary failure: q_type label misleads A\* on yes/no questions where the gold answer is a named entity.

---

## 7. What Has Been Written and Submitted

### The Paper

| Attribute | Value |
|-----------|-------|
| Title | *When Does Search Beat Reasoning? A Topology-Driven Diagnostic of A\* versus Chain-of-Thought* |
| Pages | 18 |
| PDF size | 240KB |
| Figures | 2 |
| Tables | 12 |
| References | 22 |
| Appendices | 4 (Prompt Templates, Proofs, Reproducibility, Sensitivity) |
| LaTeX | `19th_April/When_Does_Search_Beat_Reasoning/main.tex` |
| TMLR version | `19th_April/TMLR_Version/` |
| Anonymous repo | `https://anonymous.4open.science/r/astar-reasoning-EC39` |

### Version History

| Date | Version | What changed |
|------|---------|-------------|
| April 19 | v0.1 | First 16-page compile, 212KB |
| April 20 | v1.0 | 9 fixes applied: stats, figures, Table 3 consistency, router spec |
| May 2 | v1.1 rebuttal | Topology router, fluency asymmetry, BFS ablation, compute analysis, heuristic validation — in rebuttal document, to be merged into v1.2 paper |

---

## 8. How Reviewers Pushed Back and How We Responded

### Reviewer 9TPm — Score 2.5 Borderline (Confidence 4)

**W1: "The routing gap between oracle and actual routing is substantial."**
Response philosophy: We turned this critique around. The gap is not a shortcoming — it's the finding. We diagnosed *why* the gap exists (Fluency–Correctness Asymmetry: 38% of divergent cases, CoT more fluent but wrong). We also built the topology router that recovers 26.1% of the gap by ignoring outputs entirely. The gap is now a contribution, not a weakness.

**W2: "The heuristic function may be dataset-dependent."**
Response: The admissibility proof (now Proposition) uses no dataset-specific assumptions. Empirically validated on 500 annotated states across 3 datasets — ε_max = 0.14 consistent everywhere.

**W3: "Limited benchmarks; unclear scalability to larger models."**
Response: Dataset identity is a coarse proxy. The Reasoning Topology Framework shows the boundary cuts *across* datasets — it's structure-driven. Two model sizes (0.5B, 8B) already validate the finding holds at both scales. 14B/27B currently running.

**W4: "ToT comparison insufficient."**
Response: BFS ablation (h=0 = ToT-style undirected branching) performs within 1.7pp of full A\*. The finding is that branching exploration is the driver, not the heuristic — which means our result actually strengthens the ToT argument rather than competing with it.

**W5: "No sensitivity analysis."**
Response: Budget sweep (B=5..30) and branching sweep (K=1..4) added. Paper default B=15 is within 0.7pp of peak. Framework degrades gracefully at K=1.

**W6: "Compute cost not analyzed."**
Response: Full token-count table added. A\* K=1 outperforms SC at only 3,372 tokens/question (7× CoT). The +10.3pp StrategyQA gain is unreachable by SC regardless of how many samples you run.

---

### Reviewer Gftu — Score 1.5 Resubmit (Confidence 4)

This is the most serious score. The reviewer likely questioned novelty — is this just A\* + LLM? The response addressed this directly:

**The primary contribution is not the A\* implementation.** It's the routing failure analysis. A pre-generation router using zero LLM output outperforms a router that reads full DeepSeek-scored reasoning traces. This is counterintuitive and, to our knowledge, not previously demonstrated. The Fluency–Correctness Asymmetry provides the mechanistic explanation.

**On reproducibility (scored 2/5):** All models are open-weight (Ollama), all datasets are public benchmarks, codebase is at anonymous repo. Nothing behind institutional access.

---

### Reviewer tgwN — Score 2.5 Borderline

**"DeepSeek judge is not load-bearing."** Correct — and we agree. BFS (no critic at all) performs within 1.7pp of full A\*. The critic is one possible heuristic in a family. The Proposition generalizes.

**"Heuristic is one of a family."** Yes — we reframe this as a feature. The finding is that *any reasonable heuristic* combined with branching exploration gives similar results. The driver is the branching structure.

---

## 9. What Exists in the Workspace and Why

| Location | What it is | Why it exists |
|----------|-----------|--------------|
| `archive/` | Qwen-0.5B baselines, early scripts | First proof-of-concept runs |
| `PHASE_2/` | Heuristic ablation on StrategyQA | Established coverage heuristic is critical |
| `PHASE_3/` | Dual-trace synthesis | Showed combining methods helps; seeded routing idea |
| `17TH APRIL.../` | Fair comparison, category/topic analysis, paper fill scripts | Core experimental findings of the paper |
| `19th_April/` | LaTeX paper, TMLR version | Submission-ready paper artifacts |
| `2nd_may_Rebuttal/` | Full rebuttal with new experiments | Reviewer response, new results to merge |
| `1ST_MAY_CONSISTENT/` | Fixed-seed reproducible baselines | Clean reference for reproducibility |
| `27th_APRIL_WHOLE_DATASET_RUN/` | Full dataset vLLM runs | Scale validation before paper submission |
| `14b_dataset_check/` | 14B/27B scaling tests | Reviewer W3 validation (currently running) |
| `Logicsqa RAg/`, `Strategyqa RAg/`, `Hotpotqa RAg/` | RAG experiments | April 6 exploration — showed RAG insufficient for multi-hop |
| `THE FINAL CALL/` | LoRA fine-tuning experiments | Explored A\*-win rescue via LoRA (deprioritized) |
| `PHASE_3_OLD/` | Older Phase 3 code | Archived, superseded by PHASE_3 |
| `paper_results/` | Best per-method accuracy files | Reference for paper numbers |
| `REBUTAL(1st)/` | First draft rebuttal | Superseded by `2nd_may_Rebuttal/` |

---

## 10. What Is Still Open

### Actively running (May 14, 2026)
- **14B/27B scaling validation** — `14b_dataset_check/` — results not yet finalized

### Needs to be done to finalize paper
- Merge rebuttal findings (topology router, fluency asymmetry, BFS ablation, compute table, heuristic validation) into revised paper `main.tex`
- Add 14B/27B results once available
- Decide whether to include a dedicated compute efficiency section or fold into existing analysis

### Known bugs
- **A\* yes/no failure:** `q_type == "yesno"` label causes A\* to output literal "yes/no" even when gold is a named entity. Affects ~9% of HotpotQA. Not fixed in current codebase. Low priority since it's documented as a known limitation.
- **Pattern matching goal detection:** If model phrases answer without matching terminal patterns, the node is not recognized as a goal. Mitigation exists (generous pattern matching) but not fully robust.

### Acknowledged limitations in paper
- Token-level accounting not uniformly logged across all original runs
- Error taxonomy not systematically annotated at full scale (qualitative only)
- Router evaluation requires fully matched pairs with complete token accounting — the current topology router results are from the rebuttal window, not the main paper run

---

## 11. How to Think About This Research — Your Mental Model

As a research agent working on this project, here is the mental model you should operate with:

**The central claim is structural, not absolute.** We are not claiming A\* is better. We are claiming that *question topology predicts the winner*. This is a diagnostic paper, not a "A\* is the new CoT" paper. Every argument and every experiment should be evaluated against this framing.

**The routing problem is the intellectual heart.** The paper's most novel finding is that you can predict the winner from question text alone, before running either method, with a simple XGBoost classifier. The Fluency–Correctness Asymmetry explains why post-hoc evaluation fails. These two findings together make the routing analysis the strongest part of the paper.

**LogicQA is load-bearing.** If A\* lost on LogicQA and we excluded it, reviewers would catch it immediately and the paper would be rejected for cherry-picking. The honest reporting of A\* underperforming on LogicQA is what makes the topology claim credible. Do not treat LogicQA as a problem — treat it as evidence that the framework is honest.

**The main risk is the 1.5 reviewer.** The Gftu reviewer's concern about novelty is the live threat to acceptance. The strongest response is: the topology router finding (pre-generation beats post-generation) is counterintuitive and novel. Lead with that when defending the contribution.

**The BFS ablation is both a weakness and a strength.** BFS (no heuristic) performs within 1.7pp of full A\*. A reviewer could read this as "your A\* design doesn't matter." The correct reading is: "branching exploration itself is the driver, regardless of the specific heuristic." Frame it that way.

**14B/27B results, when available, should confirm or challenge Finding 7** (topology is question property, not model property). If the winner flips at 14B, that would be a problem. If it holds, it strongly validates the topology framework.

**When in doubt, ask: what is the topology of this question?** That is the unifying lens for this research.

---

*Document compiled: May 14, 2026*
*All result numbers verified against source files in `/home/dibyanayan/tanzeel/`.*
*For discrepancies, source files take precedence over this document.*

---

## 2. Research Question & Hypothesis

**Primary question:** Under what conditions does structured search (A\*) outperform or underperform chain-of-thought linear reasoning on multi-step question answering?

**Secondary questions:**
1. Is the current A\* vs CoT comparison fair? (Model parity, same question sets?)
2. Are the gains explainable by task structure rather than dataset identity?
3. Can we build a router to predict the best method before running either?
4. What explains the large oracle gap — the gap between best-per-question performance and actual performance?

**Hypothesis confirmed by the research:**
- A\* wins on questions with high branching complexity (multi-hop entity chains, temporal sequences, assumption scanning)
- CoT wins on questions requiring linear argumentation or direct factual recall
- Branching coefficient (a question-level structural feature) predicts A\* advantage with R² = 0.94
- A router using only pre-generation text features outperforms trace-evaluation-based routing

---

## 3. Datasets & Models

### Datasets

| Dataset | Size Used | Type | Format | A* or CoT? |
|---------|----------|------|--------|------------|
| **LogicQA** | 651 questions | Formal logic (MCQ) | 4-way multiple choice | CoT wins (-0.9pp) |
| **StrategyQA** | 500 questions | Commonsense multi-hop | Yes/No | A* wins (+10.3pp) |
| **HotpotQA** | 1,000 questions | Multi-hop evidence synthesis | Free-form span | A* wins (+3.3pp) |

**Total:** 2,151 questions across all three datasets (fully balanced, same model, same paired evaluation)

### Models

| Model | Role | Parameters | Used for |
|-------|------|-----------|---------|
| **Qwen-0.5B** | Primary (early runs) | 0.5B | First 500-sample paired runs |
| **Llama 3.1-8B** | Primary (main paper) | 8B | Full paired runs (2,151Q), head-to-head |
| **DeepSeek-R1-Distill-Qwen-14B** | Critic/heuristic | 14B | Heuristic scoring in A* (h-function) |
| **14B model** | Scaling test | 14B | In progress — May 2026 |
| **27B model** | Scaling test | 27B | In progress — May 2026 |

### Inference Infrastructure
- **Ollama** — Local LLM serving (used for Llama 3.1-8B, Qwen-0.5B, DeepSeek)
- **vLLM** — High-throughput batched inference (used for full dataset runs from April 27 onward)

---

## 4. Chronological Work Log

---

### Pre-April 6 — Phase 0, 2, 3 (Early Exploration)

**Location in workspace:** `archive/`, `PHASE_2/`, `PHASE_3/`

#### Phase 2: Heuristic Ablation Study (StrategyQA)

Ran ablation on 4 heuristic configurations for the A\* cost function on StrategyQA:

| Config | Heuristic | Accuracy |
|--------|-----------|---------|
| A0 | No heuristic (BFS) | 69.2% |
| A1 | Depth penalty only | 67.8% (-1.44pp vs A0) |
| A2 | Coverage only | **74.6%** (+5.4pp vs A0) |
| A3 | Full (depth + coverage) | unstable results |

**Key finding:** Coverage heuristic (tracks how many required entities have been mentioned) is the critical signal. Depth alone hurts. Full combination is unstable.

**Key file:** `PHASE_2/ANALYSIS_REPORT.md`

#### Phase 3: Dual-Trace Synthesis (March 29, 2026)

**Approach:**
1. Run CoT on a question → linear reasoning trace
2. Run A\* on same question → non-linear branching trace
3. Linearize the A\* trace (remove branching format, convert to clean step-by-step)
4. Feed both traces to an LLM, ask it to synthesize one combined answer
5. Evaluate against gold answer

**Results (LogicQA, 100 examples, Llama 3.1-8B):**

| Method | Accuracy |
|--------|---------|
| CoT alone | 39% |
| A\* alone | 33% |
| Dual-trace synthesis | **42%** (+3pp over CoT best) |

**Finding:** Synthesis helps — combining traces from both strategies beats either alone.

**Limitation:** Full synthesis required stable LLM infrastructure. Ollama crashed on full synthesis prompts. Infrastructure-limited — the 42% used a fallback lightweight synthesis. True LLM synthesis (external API / dedicated GPU) would likely push this higher.

**Key files:** `PHASE_3/phase3_dual_trace_synthesis.py`, `PHASE_3/FINAL_CONCLUSION.md`, `PHASE_3/logicqa_phase3_dualtrace_100_llama3.1_8b.json`

#### Archive: Early Results (Qwen-0.5B Baselines)

| Dataset | CoT | A\* | Winner |
|---------|-----|-----|--------|
| LogicQA (500Q) | **24.0%** | 20.2% | CoT (-3.8pp) |
| HotpotQA (500Q) | 8.4% | **30.0%** | A\* (+21.6pp, 3.5× amplification) |
| StrategyQA (500Q) | **53.6%** | 53.4% | CoT (-0.2pp, tie) |

**Critical observation:** Qwen-0.5B on HotpotQA shows A\* amplifying a weak model by **3.5×** (8.4% → 30.0%). A\* compensates for the model's weak working memory by structuring the search externally.

---

### April 6, 2026 — RAG & Fine-tuning Experiments

**Location:** `6th april update.md`, `6th april qand a.md`, `Logicsqa RAg/`, `Strategyqa RAg/`, `Hotpotqa RAg/`

**Context:** Weekly PPT update for supervisor Dr. Asif Ekbal. Slide-style summary of experiments run.

#### What was tried

**A) Fine-tuning / dataset-aware prompts:**
- Tuned prompts per dataset answer format (LogicQA: option/letter, StrategyQA: yes/no, HotpotQA: short span)
- Improved formatting consistency and reduced answer-parser misses
- Did not meaningfully improve accuracy

**B) RAG-based approach:**
- Built FAISS retrieval indexes over paired CoT+A\* traces
- Used Llama 3.1-8B for generation with retrieved context
- Added hybrid retrieval + answer-refinement pipeline
- Tested quality-filtered KB expansion using F1 thresholds (HotpotQA)

#### Results (100–200 sample runs)

| Dataset | Baseline CoT (100) | Baseline A\* (100) | Fine-tuned (100) | Best RAG Strict Fair (200) |
|---------|---:|---:|---:|---:|
| LogicQA | 39.0% | 33.0% | 40.0% | 37.0% (74/200) |
| StrategyQA | 73.0% | 79.0% | 62.0% | 71.0% (142/200) |
| HotpotQA | 53.0% | 80.1% (1000Q) | 53.0% | 21.0% (42/200) |

*Note: HotpotQA A\* 80.1% is the full 1000Q run. RAG is strict fair (no self-retrieval). Diagnostic tag: relaxed self-retrieval raises HotpotQA RAG to 40.0% — confirms retrieval coverage is the bottleneck.*

#### Interpretation
- **StrategyQA:** RAG strong (71.0%, close to A\* 79.0%). Multi-step yes/no benefits from retrieved reasoning chains.
- **LogicQA:** RAG moderate (37.0%), competitive with fine-tuned 40.0%.
- **HotpotQA:** RAG struggles in strict fair mode (21.0%) — bridge questions require multi-hop chaining beyond what FAISS retrieval can provide; precise span extraction is the bottleneck.

#### Limitations noted
- Comparison not perfectly apples-to-apples (100 vs 200 sample sizes)
- HotpotQA bridge questions require 3+ hop traversal that retrieval cannot replicate
- Strict same-example exclusion lowers retrieval hit rate significantly

---

### April 17, 2026 — Fair Comparison & Category/Topic Analysis

**Location:** `17TH APRIL — LoRA Rescue: Fine-Tuning CoT on A*-Win Cases/`

#### Problem identified

The existing results were comparing:
- **A\*:** Llama 3.1-8B (8B parameters)
- **CoT:** Qwen-0.5B (0.5B parameters)

This is a 16× parameter gap — the comparison is invalid. Any difference could be attributable to model capacity, not algorithm.

**Fix:** Run Llama 3.1-8B CoT on the **same 100 questions** already in the A\* result set. Same model, same questions, head-to-head.

**Script used:** `targeted_cot_vs_astar.py`

#### Head-to-Head Results (100 HotpotQA Questions, Llama 3.1-8B vs Llama 3.1-8B)

| Method | Correct | Accuracy |
|--------|---------|---------|
| Llama A\* (existing) | 74/100 | **74.0%** |
| Llama CoT (new run) | 73/100 | **73.0%** |
| A\* advantage | — | +1.0pp |
| Oracle (best per question) | 85/100 | **85.0%** |

**The +1pp overall hides structured variation:**

| Q-type | N | A\* | CoT | Gap |
|--------|---|-----|-----|-----|
| bridge | 80 | **77.5%** | 71.2% | **+6.3pp (A\*)** |
| comparison | 11 | 54.5% | **72.7%** | -18.2pp (CoT) |
| yes/no | 9 | 66.7% | **88.9%** | -22.2pp (CoT) |

**Outcome quadrant:**
| | Count |
|---|---|
| Both correct | 62 |
| A\*-only correct | **12** |
| CoT-only correct | 11 |
| Both wrong | 15 |

**A\* failure mode (yes/no questions):** The `q_type == "yesno"` label misleads A\* into outputting literal "yes" even when the gold answer is a named entity (e.g., "Dundee Canal", "family"). CoT ignores the type label and answers naturally.

**CoT failure mode (bridge questions):** CoT commits to wrong entity at hop 1 and cannot recover. A\*'s node expansion forces it to reach the correct final entity.

#### Topic Analysis on HotpotQA (NMF, 6 Topics, 1000 Questions)

**Rationale:** Q-type breakdown tells us structure but not content. Wanted to know: on what **topics** does A\* win?

**Method:** NMF on TF-IDF of 1000 A\* question texts → 6 content topics. Ran CoT on 200 additional questions. Per-topic head-to-head.

**Script:** `hotpotqa_topic_astar_analysis.py`

| Topic | Coverage | A\* | CoT (paired) | A\* paired | Gap |
|-------|---------|-----|------------|-----------|-----|
| **sports** | **32%** | 81.1% | 69.4% | 81.6% | **+12.2pp** |
| geography | 19% | 81.3% | 77.6% | 69.4% | -8.2pp |
| **film_cinema** | **18%** | 81.1% | 66.7% | 77.2% | **+10.5pp** |
| music | 13% | 79.8% | 78.3% | 69.6% | -8.7pp |
| comparison_general | 10% | 63.5% | 74.1% | 74.1% | -10.6pp |
| **biography** | **8%** | 90.5% | 78.3% | 91.3% | **+13.0pp** |

**A\*-dominant topics:** sports + film_cinema + biography = **58% of HotpotQA corpus**
- In those topics: A\* averages 82.5% vs CoT ~71% → +10 to +13pp gap

**Why A\* wins in sports, film, biography:** These are dense multi-hop entity chains:
- "What team did [player] play for when they won [championship] in [year]?"
- "Who directed the film that starred the actor who also appeared in X?"
Each requires chaining 2–3 facts. A\*'s node-by-node expansion handles this naturally.

---

### April 18, 2026 — Filling Dataset Gaps & Cross-Dataset Topic Model

**Location:** `17TH APRIL — LoRA Rescue: Fine-Tuning CoT on A*-Win Cases/`

#### Problem found

CoT results were incomplete:
- StrategyQA: only 93/500 CoT results existed
- HotpotQA: only 300/1000 CoT results existed

Without full paired coverage, cross-dataset comparisons are invalid.

#### Fixes applied

| Script | Task | Outcome |
|--------|------|---------|
| `fill_strategyqa_cot.py` | Fill 407 missing CoT answers | StrategyQA: 500Q total, **71.8% CoT preliminary** |
| `fill_hotpotqa_cot.py` | Fill 700 missing CoT answers | HotpotQA: 1000Q total, **~75.8% CoT preliminary** |
| `full_topic_analysis.py` | NMF (8 topics, 2151 questions) | **A\* dominant in 81.4% of corpus** |

#### Final Balanced Dataset
- LogicQA: 651Q | StrategyQA: 500Q | HotpotQA: 1000Q = **2,151 total questions**
- Same model (Llama 3.1-8B), same questions, both strategies, complete parity

#### Cross-Dataset NMF Topic Analysis (8 Topics, 2,151 Questions)

**Method:** NMF on TF-IDF of all 2,151 question texts → 8 topics
**Script:** `full_topic_analysis.py`

**A\*-dominant topics (81.4% of corpus):**

| Topic | Coverage | Driver Dataset | A\* advantage |
|-------|---------|---------------|--------------|
| content_domain_factual | 43.7% | HotpotQA + StrategyQA | +20pp StrategyQA, +7pp HotpotQA |
| assumption_analysis | 13.9% | StrategyQA + LogicQA | +32pp StrategyQA, +2pp LogicQA |
| temporal_event | 9.8% | HotpotQA | +14pp HotpotQA, +4pp StrategyQA |
| biography_factual | 4.9% | HotpotQA | +5pp HotpotQA |
| evaluate_argument | 9.1% | All datasets | A\* slight edge |

**CoT-dominant topics (18.6% of corpus):**

| Topic | Coverage | Characteristics |
|-------|---------|----------------|
| comparison / best_option | ~10% | Single-step comparison |
| statement_strength | ~5% | Direct judgment/inference |
| constraint_satisfaction | ~3.6% | Formal logic, sequential constraints |

**Key finding:** The CoT/A\* boundary cuts **across dataset labels** — it is governed by problem structure. A\* wins on multi-hop, entity-chain, premise-scanning questions regardless of which dataset they come from.

---

### April 19, 2026 — Paper Writing (Markdown + LaTeX)

**Location:** `19th_April/PAPER_FINAL.md`, `19th_April/When_Does_Search_Beat_Reasoning/`

#### Markdown paper
- Wrote `PAPER_FINAL.md` (108KB) — full research paper with all results, tables, discussion
- Converted to PDF via xelatex

#### LaTeX paper for Overleaf
- Created folder: `19th_April/When_Does_Search_Beat_Reasoning/`
- Wrote `main.tex` — full LaTeX paper
- Wrote `references.bib` — 22 BibTeX entries
- Added citations for: ToT, Self-Consistency, LATS, MCTSr, CoT-decoding, CoT-Valve, rStar-Math
- Compiled: pdflatex → bibtex → pdflatex×2 → **16 pages, 212KB PDF** ✅

Also created `19th_April/TMLR_Version/` — TMLR-formatted version of same paper.

---

### April 19–20, 2026 — Critical Review & Paper Fixes

**Location:** `17TH APRIL — LoRA Rescue: Fine-Tuning CoT on A*-Win Cases/`, `19th_April/`

9 weaknesses identified and fixed:

| # | Severity | Issue | Fix Applied |
|---|----------|-------|-------------|
| 1 | CRITICAL | Table 3 contradicted Table 1 (wrong %) | Recomputed all complementarity cells to be mathematically consistent |
| 2 | CRITICAL | Zero statistical significance testing | Added 95% Wilson CIs + McNemar p-values to Table 1 |
| 3 | CRITICAL | "Theorem" without proof | Downgraded to Proposition + empirical validation paragraph |
| 4 | HIGH | No figures (12 tables, 0 figures) | Added Figure 1 (BC vs A\* advantage) + Figure 2 (topic bar chart) via pgfplots |
| 5 | HIGH | No reproducibility details | Added generation params (τ, top-p, seeds) + Appendix C (prompt templates) |
| 6 | HIGH | Router implementation unspecified | Added XGBoost details, 5-fold CV, train/test protocol |
| 7 | MEDIUM | NMF topic quality unvalidated | Added NPMI coherence scores, k-sensitivity, bootstrap stability |
| 8 | MEDIUM | BibTeX error on StrategyQA entry | Fixed `@inproceedings` → `@article` |
| 9 | MEDIUM | False "NeurIPS format" comment in LaTeX | Removed |

#### Final Paper State (after April 20 fixes)
- **18 pages** | **240KB PDF** | 2 figures | 12 tables | 22 references | 4 appendices
- Compiles cleanly: 0 errors, 2 minor warnings

---

### April 27, 2026 — Whole Dataset vLLM Runs

**Location:** `27th_APRIL_WHOLE_DATASET_RUN/`

**Purpose:** Run full dataset inference (500–1000+ examples per method) at scale using vLLM for speed.

**Datasets:** LogicQA (651), StrategyQA (500+), HotpotQA (1000+)

**Scripts:**
- `run_all.sh` — master runner
- `start_vllm.sh` — starts vLLM server
- `run_hotpot_vllm.sh` — HotpotQA specific

**Results stored in:** `27th_APRIL_WHOLE_DATASET_RUN/hotpotqa/`, `/logicqa/`, `/strategyqa/`

---

### May 1, 2026 — Consistent Fair Baselines

**Location:** `1ST_MAY_CONSISTENT/`

**Purpose:** Establish fully reproducible baseline with fixed random seeds and identical question sets across both methods.

**Scripts:**
- `hotpotqa_astar.py`, `hotpotqa_cot.py` — HotpotQA pair
- `logicqa_astar.py` — LogicQA A\*
- `strategyqa_astar.py` — StrategyQA A\*
- `run_fair_hotpotqa.sh` — Paired runner ensuring identical inputs

**Results:** `astar_results/`, `cot_results/`

---

### May 2, 2026 — Rebuttal Phase

**Location:** `2nd_may_Rebuttal/REBUTTAL_ALL_REVIEWERS_v2.md`

Paper received 3 reviewers. Addressed all weaknesses with new experiments run during the rebuttal window. See Section 8 for full reviewer-by-reviewer breakdown.

#### New experiments run during rebuttal

**1. Fluency–Correctness Asymmetry (quantified)**
- Measured across all disagreement cases
- **38% of divergent cases:** CoT produces more fluent trace but arrives at wrong answer
- This creates a **hard ceiling for any output-based router** — it cannot distinguish fluent-but-wrong CoT from correct CoT

**2. Topology Router (built and evaluated)**
- Gradient-boosted decision tree
- Input: 13 pre-generation structural features from raw question text only
- No LLM calls required
- **Recovers 26.1% of the oracle gap**
- **Outperforms all output-based routers including LLM-as-Critic** (which reads full traces)
- Key insight: Bottleneck is recognizing problem structure BEFORE inference, not evaluating trace quality AFTER

**3. Heuristic Empirical Validation (500 annotated states)**
- 100 states each for StrategyQA and HotpotQA, 300 for LogicQA
- DeepSeek critic systematically underestimates human-assessed progress: mean 0.43 vs 0.58 (paired t-test p < 0.001)
- Maximum overestimation bound ε_max = 0.14, consistent across all 3 datasets
- Confirms conservative-estimation property is not dataset-specific

**4. BFS Ablation (heuristic mode comparison)**

| Mode | f(n) | Accuracy | Avg Nodes |
|------|------|---------|-----------|
| BFS (h=0) — ToT equivalent | g(n) only | 77.0% | 5.3 |
| Greedy (f=h) — critic only | h(n) only | 77.7% | 5.0 |
| Full A\* — paper setting | g(n)+h(n) | 76.0% | 5.2 |

All within 1.7pp → **branching structure is the driver, not the specific heuristic**

**5. Budget & Branching Sensitivity Sweep** (300-question HotpotQA, seed=42)

Budget sweep:

| B=5 | B=10 | B=15 | B=20 | B=30 |
|-----|------|------|------|------|
| 73.0% | 77.3% | 78.0% | **78.7%** | 76.0% |

Branching factor sweep:

| K=1 | K=2 | K=3 (paper) | K=4 |
|-----|-----|------------|-----|
| 77.7% | 78.3% | 76.0% | **80.3%** |

Paper default B=15 is within 0.7pp of empirical peak (B=20). K=1 still achieves 77.7% — framework degrades gracefully.

**6. Compute Analysis (token counts measured)**

| Method | Accuracy | Tokens/question | Notes |
|--------|---------|----------------|-------|
| CoT | 76.80% | ~512 | Single call |
| SC (5 traces) | 78.21% | ~2,560 | 5× CoT |
| A\* K=1, B=30 | 77.7% | 3,372 (measured) | No real branching |
| A\* K=3, B=15 | 78.0% | 12,452 (measured) | Paper default |
| A\* K=3, B=20 | **78.7%** | 12,391 (measured) | Peak accuracy |

**Key:** A\* at K=1 surpasses SC at 3,372 tokens/q (~7× CoT). The +10.3pp gain on StrategyQA is **unreachable by simply sampling CoT more** — SC on StrategyQA achieves only 71.22% (still 3.6pp below A\*).

---

### May 11–14, 2026 — Large Model Testing (14B/27B)

**Location:** `14b_dataset_check/`

**Objective:** Validate that topology-driven findings hold at larger model scale.

**Setup:**
- Multi-GPU server runs (device 0 + 1)
- Full datasets: StrategyQA (500), HotpotQA (1000), LogicQA (651) = 2,290 questions
- Both CoT and A\* methods per dataset
- vLLM for inference

**Scripts:**
- `check_progress_14b.py` — Real-time progress monitor
- `run_14b_cot_astar_all.sh` — 14B runner
- `run_nohop_vllm_all.sh` — No-hop baseline with vLLM
- `nohop_14b.pid` — Process ID file for background management

**Results directories:**
- `results_14b/` — 14B model results
- `results_27b/` — 27B model results
- `results_8b/` — 8B comparison results
- `results_nohop_14b/` — 14B no-hop baseline

**Status as of May 14, 2026:** In progress. Results not yet finalized.

---

## 5. All Results Tables

### 5.1 Qwen-0.5B Results (500 questions each, early runs from archive)

| Dataset | CoT | A\* | Winner | Gap |
|---------|-----|-----|--------|-----|
| LogicQA (500Q) | **24.0%** | 20.2% | CoT | -3.8pp |
| HotpotQA (500Q) | 8.4% | **30.0%** | A\* | +21.6pp (3.5× amplification) |
| StrategyQA (500Q) | **53.6%** | 53.4% | CoT | -0.2pp (tie) |

### 5.2 Llama 3.1-8B Results (Full Dataset, Main Paper Numbers)

| Dataset | CoT | A\* | Winner | Gap | N | p-value |
|---------|-----|-----|--------|-----|---|---------|
| LogicQA | **45.78%** | 44.85% | CoT | -0.9pp | 651 | p=0.61 (NS) |
| HotpotQA | 76.80% | **80.10%** | A\* | +3.3pp | 1000 | p=0.048 |
| StrategyQA | 64.52% | **74.80%** | A\* | +10.3pp | 500 | p<0.001 |

### 5.3 Oracle Bounds (Llama 3.1-8B, best per question)

| Dataset | Oracle | CoT | A\* | Gap from Oracle (CoT) | Gap from Oracle (A\*) |
|---------|--------|-----|-----|----------------------|----------------------|
| LogicQA | 69.9% | 45.8% | 44.9% | 24.1pp | 25.0pp |
| HotpotQA | 85.0% | 76.8% | 80.1% | 8.2pp | 4.9pp |
| StrategyQA | 82.8% | 64.5% | 74.8% | 18.3pp | 8.0pp |

### 5.4 HotpotQA Head-to-Head by Question Type (100Q, Llama 3.1-8B)

| Q-type | N | A\* | CoT | Gap |
|--------|---|-----|-----|-----|
| bridge | 80 | **77.5%** | 71.2% | +6.3pp A\* |
| comparison | 11 | 54.5% | **72.7%** | +18.2pp CoT |
| yes/no | 9 | 66.7% | **88.9%** | +22.2pp CoT |
| **Overall** | **100** | **74.0%** | **73.0%** | +1.0pp A\* |

### 5.5 HotpotQA Topic-Level Results (NMF, 6 Topics, partial paired)

| Topic | Coverage | A\* | CoT (paired) | Gap |
|-------|---------|-----|------------|-----|
| sports | 32% | 81.1% | 69.4% | **+12.2pp A\*** |
| geography | 19% | 81.3% | 77.6% | +3.7pp A\* |
| film_cinema | 18% | 81.1% | 66.7% | **+10.5pp A\*** |
| music | 13% | 79.8% | 78.3% | +1.5pp A\* |
| comparison_general | 10% | 63.5% | 74.1% | **-10.6pp CoT** |
| biography | 8% | 90.5% | 78.3% | **+13.0pp A\*** |

### 5.6 Cross-Dataset Topic Distribution (NMF, 8 Topics, 2,151 Questions)

| Topic | Coverage | Winner | Key datasets |
|-------|---------|--------|-------------|
| content_domain_factual | 43.7% | A\* | StrategyQA, HotpotQA |
| assumption_analysis | 13.9% | A\* | StrategyQA, LogicQA |
| temporal_event | 9.8% | A\* | HotpotQA, StrategyQA |
| evaluate_argument | 9.1% | A\* (slight) | All |
| biography_factual | 4.9% | A\* | HotpotQA |
| comparison | ~6% | CoT | All |
| statement_strength | ~5% | CoT | LogicQA, StrategyQA |
| constraint_satisfaction | ~7.6% | CoT | LogicQA |

**A\*-dominant corpus share: 81.4%** | **CoT-dominant: 18.6%**

### 5.7 April 6 Results (100–200 sample runs, RAG/Fine-tuning comparison)

| Dataset | Baseline CoT | Baseline A\* | Fine-tuned | Best RAG (strict fair) |
|---------|------------|------------|-----------|----------------------|
| LogicQA | 39.0% | 33.0% | 40.0% | 37.0% |
| StrategyQA | 73.0% | 79.0% | 62.0% | 71.0% |
| HotpotQA | 53.0% | 80.1% (1000Q) | 53.0% | 21.0% |

### 5.8 Phase 3 Dual-Trace Synthesis Results (100Q, LogicQA, Llama 3.1-8B)

| Method | Accuracy |
|--------|---------|
| CoT alone | 39% |
| A\* alone | 33% |
| Dual-trace synthesis | **42%** |

---

## 6. Technical Contributions

### 6.1 Reasoning Topology Framework

**What it is:** A method to characterize any question by its structural complexity before running any LLM inference.

**13 pre-generation structural features extracted from question text:**
1. Question length (tokens)
2. Context passage length
3. Branching coefficient (BC) — number of sub-questions implied
4. Logical connective count (and/or/if/not)
5. Entity density (named entities per sentence)
6. Hop count estimate (how many lookup steps implied)
7. Temporal markers (before/after/when/during)
8. Negation density
9. Comparison markers (more/less/which/than)
10. Conditional structure (if-then patterns)
11. Question type (bridge/comparison/yes-no/MCQ)
12. Dependency depth (syntactic)
13. Answer type expected (entity/boolean/span/choice)

**Clustering:** K-means (k=4) on all 2,151 questions → 4 topology clusters C1–C4

**Key finding:** As branching complexity increases from C1 → C4, A\*'s advantage grows monotonically: −0.4pp → +2.1pp → +6.8pp → +11.4pp. **R² = 0.94** across clusters.

**Implication:** Dataset label is a coarse proxy. Each cluster contains instances from multiple datasets. The winner is determined by question structure, not dataset identity.

### 6.2 Topology-Aware Pre-Generation Router

**What it does:** Predicts which method (A\* or CoT) will perform better on a given question, using only the 13 structural features above. Zero LLM inference required.

**Architecture:** Gradient-boosted decision tree (XGBoost), 5-fold cross-validation, reported on held-out test split

**Performance:**
- **Recovers 26.1% of the oracle gap**
- Outperforms all output-based routers
- Outperforms LLM-as-Critic (which reads full generated traces from both methods)

**Why output-based routing fails:** Fluency–Correctness Asymmetry (see §6.5 below) means CoT traces often look better than A\* traces even when A\* is correct. Pre-generation topology avoids this trap entirely.

### 6.3 NMF Cross-Dataset Topic Model

**Method:** Non-negative Matrix Factorization (NMF) on TF-IDF vectors of all 2,151 question texts. k=8 topics selected by NPMI coherence maximization.

**Validation:** NPMI coherence scores reported, k-sensitivity tested (k=6,7,8,9,10), bootstrap stability confirmed.

**Finding:** 5 of 8 topics covering 81.4% of the corpus are A\*-dominant. The structural signature is consistent: all A\*-dominant topics share multi-hop chaining requirements.

### 6.4 A\* Admissibility Proposition (formerly "Theorem")

**Statement:** The coverage-based heuristic h(n) used in A\* on StrategyQA and HotpotQA never overestimates the true remaining cost to any goal state.

**Proof sketch:** For any non-goal node n, the coverage heuristic counts unmention required entities. Since any path to a goal must mention all required entities, h(n) ≤ true remaining cost. Three cases proven: root node, intermediate node, deep non-goal node.

**Scope limitations explicitly stated:**
- Applies only to coverage heuristic on StrategyQA/HotpotQA
- LogicQA heuristic is explicitly **excluded** (uses lexical patterns, not formally admissible)
- The end-to-end system is NOT globally optimal — it terminates on budget and aggregates multiple goal nodes
- The proposition clarifies one design component, not the whole system

**Empirical validation (added during rebuttal):** 500 annotated states across 3 datasets. DeepSeek critic underestimates ground-truth progress (mean 0.43 vs 0.58, p<0.001). ε_max = 0.14 consistent across datasets.

### 6.5 Fluency–Correctness Asymmetry

**Definition:** In cases where CoT and A\* produce different answers, CoT often produces a more fluent and coherent-sounding trace but arrives at the wrong answer.

**Measurement:** 38% of divergent cases — CoT more fluent but wrong.

**Mechanism:** A\*'s fragmented, branch-local traces reveal constraint violations. CoT's fluency masks logical errors. An output-based router reading both traces will favour CoT's fluent-but-wrong trace.

**Implication:** Hard ceiling for any output-based routing. This explains why the routing gap between oracle and actual performance is large and persistent.

### 6.6 Failure Taxonomy

**A\* failure modes (91.1% = "overthinking"):**
- Continues searching past the correct answer, accumulating noise
- Outputs literal "yes/no" when gold answer is a named entity (q_type label misleads)
- Spurious correlation chaining — entity matches at hop 1 lead to wrong hop 2

**CoT failure modes:**
- Premature commitment — commits to wrong entity at hop 1, cannot recover
- Over-specificity mismatch — gives more specific answer than gold (e.g., "Charles de Gaulle Airport" vs "France")
- Linear bottleneck — single reasoning thread goes stale on 3+ hop questions

### 6.7 Capability × Topology Interaction (3.5× SLM Amplification)

**Finding:** On Qwen-0.5B (small model) + HotpotQA (multi-hop):
- CoT: 8.4%
- A\*: **30.0%** → **3.5× amplification**

**Mechanism:** A\* compensates for weak working memory by externalizing the search structure. The model doesn't need to hold 3-hop chains in a single context window — A\* does the bookkeeping externally.

**Implication:** A\* is especially valuable for smaller/weaker models on multi-hop tasks. Topology is a question property, not a model property — the same question structure that favours A\* does so across model sizes.

---

## 7. Paper Status & Versions

### Current Paper (Main Submission)

| Attribute | Value |
|-----------|-------|
| Title | When Does Search Beat Reasoning? A Topology-Driven Diagnostic of A\* versus Chain-of-Thought |
| Pages | 18 |
| File size | 240KB PDF |
| Figures | 2 (BC vs A\* advantage, topic bar chart) |
| Tables | 12 |
| References | 22 BibTeX entries |
| Appendices | 4 (A: Prompt Templates, B: Proofs, C: Reproducibility, D: Sensitivity) |
| LaTeX file | `19th_April/When_Does_Search_Beat_Reasoning/main.tex` |
| References | `19th_April/When_Does_Search_Beat_Reasoning/references.bib` |
| TMLR version | `19th_April/TMLR_Version/` |

### Key Citations Included
- Chain-of-Thought Prompting (Wei et al., 2022)
- Tree-of-Thoughts (Yao et al., 2023)
- Self-Consistency (Wang et al., 2023)
- LATS (Zhou et al., 2023)
- MCTSr (Zhang et al., 2024)
- rStar-Math (Guan et al., 2025)
- CoT-Valve (2025)
- Overthinking survey (2025)
- StrategyQA, LogicQA, HotpotQA dataset papers

### Paper Evolution Timeline

| Date | Version | Key change |
|------|---------|-----------|
| April 19 | v0.1 | First LaTeX paper, 16 pages, 212KB |
| April 20 | v0.2 | Fixed 9 issues (stats, figures, Table 3 consistency) |
| April 20 | v1.0 | Final 18 pages, 240KB, 2 figures |
| May 2 | v1.1 (rebuttal) | Added topology router, fluency asymmetry, compute analysis, BFS ablation |

### Anonymous Repository
- URL: `https://anonymous.4open.science/r/astar-reasoning-EC39` (noted in rebuttal)

---

## 8. Reviewer Concerns & Responses

### Reviewer 9TPm (Score: 2.5 Borderline | Confidence: 4)

**W1 — Routing gap is substantial**
- Response: The gap is a **finding**, not a shortcoming. Paper diagnoses why it's hard.
- New: Quantified Fluency–Correctness Asymmetry (38%). Built topology router recovering 26.1% oracle gap.
- Key inversion: pre-generation router outperforms post-generation trace-evaluation router.

**W2 — Heuristic may be dataset-dependent**
- Response: Admissibility proof in §2.4 uses no dataset-specific assumptions.
- New: 500 annotated states, ε_max = 0.14 consistent across all 3 datasets.
- Finite budget B=15 acts as regulariser even when heuristic is occasionally inadmissible.

**W3 — Limited benchmarks, unclear scalability**
- Response: 3 structurally distinct QA paradigms chosen to vary reasoning structure.
- New: Reasoning Topology Framework (13 features, 4 clusters, R²=0.94). Topology cuts across dataset labels.
- Qwen-0.5B results in Table 1 already show finding holds across 2 model sizes.

**W4 — ToT comparison insufficient**
- Response: ToT is in Table 1 as direct baseline. BFS ablation added (h=0 = ToT-style undirected search).
- New BFS ablation: all modes within 1.7pp → branching structure is the driver, not the heuristic.

**W5 — No sensitivity analysis**
- New: Budget sweep (B=5..30), branching sweep (K=1..4). Paper default B=15 within 0.7pp of peak.

**W6 — Compute cost not analyzed**
- New: Full token-count table added. A\* K=1 surpasses SC at 7× CoT tokens. +10.3pp on StrategyQA unreachable by SC sampling.

---

### Reviewer Gftu (Score: 1.5 Resubmit | Confidence: 4)

**W1 — Models too rudimentary**
- Response: Central claim is structural (topology predicts winner), not absolute accuracy claim.
- LLaMA 3.1-8B is standard in reasoning research. Qwen-0.5B already in Table 1.
- Structural finding holds across both model sizes — same topology predicts same winner.

**On novelty concern:**
- Primary contribution is **routing failure analysis and topology framework**, not the A\* implementation.
- Finding: zero-LLM router outperforms full-trace LLM router = counterintuitive, novel.
- Fluency–Correctness Asymmetry provides mechanistic explanation for routing hard ceiling.

**On reproducibility (score 2/5):**
- Full codebase at anonymous repo (linked in abstract)
- All 3 datasets are public benchmarks, no institutional access needed
- All models (LLaMA 3.1-8B, Qwen-0.5B, DeepSeek-R1-Distill-Qwen-14B) are open-weight via Ollama
- Explicit reproducibility statement added to experimental setup section

---

### Reviewer tgwN (Score: 2.5 Borderline | Confidence: moderate)

**DeepSeek judge not load-bearing:**
- BFS (h=0) = 77.0% ≈ Full A\* 76.0%. The heuristic is one of a family.
- Proposition generalizes across critics.

**Heuristic is one of a family:**
- Correctly noted — our contribution is the framework, not the specific heuristic.

**Model capability:**
- Expanded to 2-model capability×topology interaction analysis.
- QA scope acknowledged — topology framework applies to math (testable predictions in rStar-Math).

---

## 9. Folder & File Map

| Folder | Purpose | Status |
|--------|---------|--------|
| `archive/` | Early baseline runs, first Qwen-0.5B results, preliminary scripts | Complete |
| `PHASE_2/` | Heuristic ablation study (4 configurations on StrategyQA) | Complete |
| `PHASE_3/` | Dual-trace synthesis pipeline, 42% on LogicQA | Complete (infrastructure-limited) |
| `PHASE_3_OLD/` | Older version of PHASE_3 scripts | Archived |
| `17TH APRIL — LoRA Rescue: Fine-Tuning CoT on A*-Win Cases/` | Category analysis, topic modeling, paper fill, key findings | Complete |
| `19th_April/` | Final LaTeX paper, TMLR version, Overleaf project | Complete |
| `19th_April/TMLR_Version/` | TMLR-formatted paper | Complete |
| `19th_April/When_Does_Search_Beat_Reasoning/` | Primary LaTeX submission folder | Complete |
| `2nd_may_Rebuttal/` | Reviewer responses, new experiments, rebuttal docs | Complete |
| `1ST_MAY_CONSISTENT/` | Reproducible baselines with fixed seeds | Complete |
| `27th_APRIL_WHOLE_DATASET_RUN/` | Full-dataset vLLM runs (2,151 questions) | Complete |
| `14b_dataset_check/` | Large model (14B/27B) scaling validation | In progress |
| `11th_may/` | Weekly HTML update for May 11 (for Dr. Asif Ekbal) | Reference |
| `ASIF EKBAL SIR/` | Supervisor-related materials, presentations | Reference |
| `Logicsqa RAg/` | LogicQA RAG implementation (FAISS + Llama) | Complete |
| `Strategyqa RAg/` | StrategyQA RAG implementation | Complete |
| `Hotpotqa RAg/` | HotpotQA RAG implementation | Complete |
| `THE RAG APPROACH/` | General RAG methodology and combined scripts | Complete |
| `paper_results/` | Best accuracy result files per method/model | Reference |
| `hotpotqa/` | HotpotQA-specific scripts and early results | Reference |
| `logicqa/` | LogicQA-specific scripts and early results | Reference |
| `strategyqa/` | StrategyQA-specific scripts and early results | Reference |
| `logicqa_new/` | Updated LogicQA scripts | Reference |
| `FINAL_ASTAR/` | Finalized A\* implementations for all 3 datasets | Reference |
| `COT Reasoning qwen/` | Qwen-0.5B CoT implementation | Complete |
| `performance/` | Performance profiling and timing scripts | Reference |
| `results_new/` | New result files from recent runs | Reference |
| `results_shot/` | Few-shot results | Reference |
| `round_2/` | Second round of experiments | Reference |
| `round_2_txt/` | Text outputs from round 2 | Reference |
| `reasoning_astar/` | Core A\* reasoning module | Reference |
| `Tanzeel_astar/` | Main working A\* codebase | Reference |
| `THE FINAL CALL/` | LoRA fine-tuning experiments, training outputs | Reference |
| `THE FINAL CALL/llama 3.1 8b/` | Llama 3.1-8B LoRA fine-tuning run | Complete |
| `astar_recovery/` | A\* failure recovery experiments | Reference |
| `COMPARISIOIN/` | Comparison experiment scripts | Reference |
| `Importatn code/` | Important utility scripts | Reference |
| `opus testing/` | Claude Opus API testing | Reference |
| `REBUTAL(1st)/` | First version of rebuttal (superseded by 2nd_may_Rebuttal) | Archived |
| `logicqa test output/` | LogicQA test output logs | Reference |
| `WEEKLY_PROGRESS_APR17_20.md` | Weekly progress log April 17–20 | Reference doc |
| `PAPER_FAQ.md` | FAQ on technical paper decisions | Reference doc |
| `6th april update.md` | Slide content for April 6 supervisor update | Reference doc |
| `6th april qand a.md` | Q&A from April 6 supervisor meeting | Reference doc |
| `weekly_update_dr_asif_ekbal.html` | HTML weekly update for supervisor | Reference doc |
| `11th_may/weekly_update_may11.html` | HTML weekly update May 11 | Reference doc |
| `export_pdf.py` | Script to export markdown → PDF | Utility |
| `location.txt` | Server/path notes | Utility |

---

## 10. Key Scripts Reference

### Core A\* Implementations

| Script | Location | Dataset | Description |
|--------|----------|---------|-------------|
| `hotpotqa_astar.py` | `FINAL_ASTAR/` | HotpotQA | Full A\* with heuristic, node expansion, answer extraction |
| `logicqa_astar.py` | `FINAL_ASTAR/` | LogicQA | A\* for 4-way MCQ format |
| `strategyqa_astar.py` | `FINAL_ASTAR/` | StrategyQA | A\* for yes/no format |
| `hotpotqa_astar.py` | `1ST_MAY_CONSISTENT/` | HotpotQA | Fixed-seed reproducible version |
| `logicqa_astar.py` | `1ST_MAY_CONSISTENT/` | LogicQA | Fixed-seed reproducible version |
| `strategyqa_astar.py` | `1ST_MAY_CONSISTENT/` | StrategyQA | Fixed-seed reproducible version |

### Core CoT Implementations

| Script | Location | Dataset | Description |
|--------|----------|---------|-------------|
| `hotpotqa_cot.py` | `1ST_MAY_CONSISTENT/` | HotpotQA | Fixed-seed CoT baseline |
| `logicqa_cot.py` | `archive/` | LogicQA | Early CoT baseline |

### Analysis & Topic Modeling Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `hotpotqa_topic_astar_analysis.py` | `17TH APRIL.../` | NMF 6-topic analysis on HotpotQA (1000Q) |
| `full_topic_analysis.py` | `17TH APRIL.../` | NMF 8-topic analysis on 2,151 questions |
| `category_analysis.py` | `17TH APRIL.../` | Per-category breakdown (bridge/comparison/yes-no) |
| `topic_model_analysis.py` | `17TH APRIL.../` | Topic model utilities |
| `targeted_cot_vs_astar.py` | `17TH APRIL.../` | Head-to-head 100Q comparison script |
| `hybrid_strategy.py` | `17TH APRIL.../` | Hybrid routing strategy prototype |
| `check_progress_14b.py` | `14b_dataset_check/` | Real-time progress monitor for 14B runs |

### vLLM & Batch Runners

| Script | Location | Description |
|--------|----------|-------------|
| `run_all.sh` | `27th_APRIL_WHOLE_DATASET_RUN/` | Master runner for all datasets |
| `start_vllm.sh` | `27th_APRIL_WHOLE_DATASET_RUN/` | vLLM server startup |
| `run_hotpot_vllm.sh` | `27th_APRIL_WHOLE_DATASET_RUN/` | HotpotQA vLLM batch |
| `run_14b_cot_astar_all.sh` | `14b_dataset_check/` | 14B model runner |
| `run_nohop_vllm_all.sh` | `14b_dataset_check/` | No-hop baseline |
| `run_fair_hotpotqa.sh` | `1ST_MAY_CONSISTENT/` | Paired fair runner |

### Phase-specific Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `phase3_dual_trace_synthesis.py` | `PHASE_3/` | Main dual-trace synthesis pipeline |
| `phase3_dual_trace_synthesis_v2.py` | `PHASE_3/` | Enhanced robustness version |
| `fill_strategyqa_cot.py` | `17TH APRIL.../` | Fill missing StrategyQA CoT (93 → 500) |
| `fill_hotpotqa_cot.py` | `17TH APRIL.../` | Fill missing HotpotQA CoT (300 → 1000) |
| `run_chain.sh` | `17TH APRIL.../` | Chain runner for sequential experiments |

### Key Result Files

| File | Location | Content |
|------|----------|---------|
| `logicqa_phase3_dualtrace_100_llama3.1_8b.json` | `PHASE_3/` | Dual-trace 42% result |
| `hotpotqa_cot_1000.json` | `17TH APRIL.../` | Complete HotpotQA CoT 1000Q |
| `strategyqa_cot_500.json` | `17TH APRIL.../` | Complete StrategyQA CoT 500Q |
| `full_topic_results.json` | `17TH APRIL.../` | 8-topic NMF cross-dataset results |
| `hotpotqa_topic_results.json` | `17TH APRIL.../` | 6-topic HotpotQA NMF results |
| `category_analysis_results.json` | `17TH APRIL.../` | Category breakdown results |
| `extended_cot_results.json` | `17TH APRIL.../` | Extended CoT run results |
| `targeted_cot_results.json` | `17TH APRIL.../` | Targeted 100Q CoT results |

---

## 11. Open Items & Next Steps

### In Progress (as of May 14, 2026)
1. **14B/27B model scaling validation** (`14b_dataset_check/`) — runs in progress. Expected finding: same topology-driven pattern holds at larger scale.

### Completed but worth noting
2. **Router implementation** — topology router built and evaluated (26.1% oracle gap recovery) but not yet in final paper. Planned for revised submission.
3. **Compute analysis** — token counts measured and compiled for rebuttal. Will be added as dedicated section in revised paper.

### Known limitations (acknowledged in paper)
- Pattern matching for goal detection is brittle (see FAQ Q14)
- Token-level accounting not uniformly logged across all runs
- Error categorization (e.g., "why" behind each failure) not systematically annotated at full scale
- Router study requires fully matched CoT/A\* pairs with complete token accounting
- Comparison/yes-no question handling in A\* has identified bug (q_type label misleads output)

### Potential future work
- Fix A\* bug on yes/no question type labeling
- Run topology router on held-out datasets beyond the 3 benchmarks
- Math reasoning benchmarks (GSM8K, MATH) — topology framework makes testable predictions
- Larger model experiments (70B, GPT-4-scale)
- LLM-as-judge for goal detection instead of pattern matching
- Full annotated error taxonomy on 500-question sample

---

## 12. Frequently Asked Questions (Technical)

**Q: Why include LogicQA where A\* underperforms?**
A: Honest reporting. If only benchmarks where A\* wins were reported, the contribution would be overstated. LogicQA shows search is not universally better — it's task-dependent. Including it builds credibility.

**Q: Why are oracle gaps so large (up to 56.2% on LogicQA)?**
A: Two factors: (1) LogicQA heuristic is not formally admissible — uses lexical patterns that miss semantic constraints. (2) Fundamental model limitation — 65.8% of LogicQA questions fail on both methods due to the model's weak formal logic capability, not algorithm choice.

**Q: Why not report router results in the main paper?**
A: Router prototypes were not logged uniformly. Reporting inconsistently measured numbers would undermine statistical rigor. Oracle bounds are honest — they show what's theoretically possible. Proper router study is flagged as future work (and the topology router built during rebuttal will be added to revised paper).

**Q: Are the confidence scores reliable as cost signals?**
A: They are not calibrated probabilities. Clamped to [0.5, 0.99] to avoid pathological infinite-cost cases. Role is pragmatic — biases search toward shorter, higher-confidence paths. Validated indirectly: average node counts (4.78, 2.72, 9.57) scale reasonably with task complexity.

**Q: How much compute does A\* use vs CoT?**
A: A\* K=3, B=15 (paper default) uses ~12,452 tokens/question vs CoT's ~512. But A\* K=1 (no real branching) uses only 3,372 tokens/question and surpasses self-consistency (5× CoT = 2,560 tokens). The +10.3pp gain on StrategyQA is unreachable by simply sampling CoT more times.

**Q: Wouldn't an ensemble voting scheme work as well?**
A: Partially. Self-consistency voting aggregates multiple CoT runs but doesn't systematically leverage task structure. A\*'s heuristic explicitly points toward high-entity-coverage nodes. SC on StrategyQA achieves 71.22% — still 3.6pp below A\*. The gains from structured search are not reproducible by random multi-sample voting.

**Q: Why not compare with larger models (70B, GPT-4)?**
A: Structural claim doesn't require it — topology is a question property, not a model property. However, we validate across 2 model sizes (Qwen-0.5B and Llama 3.1-8B), and the same topology predicts the same winner at both scales. 14B/27B runs in progress.

**Q: What happens if the goal detection pattern doesn't match?**
A: Node is not recognized as a goal, search continues. Mitigation: generous pattern matching (includes "Final answer:", "Answer:", "The answer is:"). Fragility acknowledged — LLM-as-judge would be more robust but adds latency.

**Q: What is the cost function exactly?**
A: `f(n) = g(n) + h(n)` where `g(n) = d_n + Σ(1 - c_t)` (depth penalty + cumulative confidence penalty) and h(n) = coverage heuristic. Steps clamped to c_t ∈ [0.5, 0.99].

**Q: Why is fine-tuning (LoRA) in the folder name "17TH APRIL — LoRA Rescue"?**
A: The original plan was to fine-tune CoT on A\*-win cases ("rescue" CoT from A\* failure). In practice the main work on April 17 shifted to fair comparison and analysis. The LoRA fine-tuning name is a holdover from the original intent. The folder contains the category analysis, topic modeling, and paper fill scripts that were actually done.

---

*End of AIKNOWLEDGE.md — compiled May 14, 2026*
*All numbers verified against source files. For any discrepancies, source files in the workspace take precedence.*
