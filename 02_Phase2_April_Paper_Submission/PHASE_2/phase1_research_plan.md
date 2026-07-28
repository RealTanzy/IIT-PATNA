# Phase 2 Research Extension Plan (Evidence-Driven)

## 1) Why We Are Extending Now

Phase 1 established that A* search can substantially improve reasoning in specific regimes, but gains are uneven across datasets and models.

This phase upgrades the work from "method demo" to "paper-grade evidence":
- isolate where A* helps,
- isolate where it hurts,
- separate reasoning gains from compute effects,
- design a robust hybrid policy.

## 2) Ground Truth From Current Repo Results

The extension plan is anchored to existing findings in this repository.

### Core empirical signals
- HotpotQA (Llama-3.1-8B): A* reached about 77-80% while earlier CoT baseline was much lower in archived runs.
- HotpotQA (Qwen-0.5B): A* (30.0%) strongly outperformed CoT (8.4%), showing search helps weak models on multi-hop QA.
- LogicQA (Llama-3.1-8B): CoT (45.8%) slightly beats A* (44.9%), so A* is not universally better.
- StrategyQA disagreement analysis (93-item subset): CoT majority beat A* majority (59.1% vs 52.7%) in 3-run voting.
- There are hard questions where both methods fail consistently, suggesting factual knowledge limits rather than search quality alone.

### Interpretation
- A* seems strongest on multi-hop compositional reasoning.
- CoT remains competitive or better in short-form binary inference and some logic formats.
- Variance and voting behavior matter as much as raw single-run accuracy.

## 3) Phase 2 Objectives

1. Quantify the source of gains: search quality vs prompting vs extra compute.
2. Build a fair evaluation protocol with budget parity and confidence intervals.
3. Develop a practical hybrid strategy (A*, CoT, and optional ensemble/router).
4. Reduce known failure classes using targeted interventions, not broad retuning.

## 4) Prioritized Experiment Tracks

## Track A: Heuristic Ablation (Highest Priority)

Question:
- Which A* components actually drive gains?

Design:
- Compare variants on same splits and model:
  - A0: no heuristic (uniform or FIFO expansion)
  - A1: depth-only heuristic
  - A2: coverage-only heuristic
  - A3: full heuristic (current best)

Success criteria:
- clear ordering by accuracy and stability,
- report delta accuracy, token cost, node expansions.

## Track B: Budget-Matched Fairness Study

Question:
- Are A* gains due to better reasoning or simply more compute/token budget?

Design:
- Equalize by total tokens/question or API calls/question.
- Compare:
  - CoT single,
  - CoT self-consistency N,
  - A* single,
  - A* self-consistency N.

Success criteria:
- performance curves under matched budgets,
- identify the Pareto frontier (accuracy vs cost).

## Track C: Failure Taxonomy and Recovery

Question:
- Which failures are recoverable by reruns/search and which are knowledge ceiling failures?

Design:
- Reuse recovery labels (never/rare/often/always) on failure subsets.
- Annotate representative errors into categories:
  - factual missing knowledge,
  - decomposition error,
  - wrong branch commitment,
  - extraction/format error.

Success criteria:
- category distribution,
- per-category recovery rates,
- intervention map by category.

## Track D: Cross-Dataset Transfer

Question:
- Why does HotpotQA gain strongly while LogicQA does not?

Design:
- Port best Hotpot-style heuristic settings to LogicQA and StrategyQA with controlled tuning windows.
- Run minimal sensitivity sweep over branching factor, max depth, and node cap.

Success criteria:
- either measurable transfer gains,
- or strong evidence that heuristic policy is dataset-specific.

## Track E: Hybrid Decision Policy

Question:
- Can we route questions to CoT or A* to exceed either standalone method?

Design:
- Build lightweight router features from question text:
  - length,
  - entity count,
  - multi-hop cues,
  - temporal/comparison markers.
- Evaluate:
  - static rules,
  - confidence-based fallback,
  - two-model vote.

Success criteria:
- hybrid beats best standalone under same budget.

## 5) Immediate Code Touchpoints

Primary code and reports for this phase:
- performance/experiment.py
- performance/run3.py
- performance/majority_vote.py
- performance/analysis_report.txt
- FINAL_ASTAR/hotpotqa_astar.py
- FINAL_ASTAR/logicqa_astar.py
- FINAL_ASTAR/strategyqa_astar.py
- astar_recovery/astar_recovery_experiment.py

## 6) Evaluation Protocol (Paper-Grade)

Mandatory controls for all new runs:
- fixed model version and run metadata,
- fixed prompt template versioning,
- odd-number vote counts for majority analyses,
- confidence intervals (bootstrap or binomial),
- explicit token/call cost reporting,
- no claims from disagreement-only subsets without full-set confirmation.

Required metrics table for each experiment:
- accuracy,
- F1 where applicable,
- average nodes expanded,
- average tokens/question,
- runtime/question,
- variance across seeds.

## 7) Execution Plan (2 Weeks)

Week 1:
1. Implement ablations and budget-matched runner.
2. Reproduce baseline numbers with pinned configs.
3. Produce Track A + Track B result tables.

Week 2:
1. Run failure taxonomy and recovery pipeline.
2. Run cross-dataset transfer checks.
3. Build and test simple hybrid router.
4. Draft final narrative with claims limited to supported evidence.

## 8) Expected Research Contribution After Phase 2

If completed successfully, Phase 2 can claim:
- when search helps and why,
- when CoT remains superior and why,
- compute-aware fairness rather than raw-accuracy-only claims,
- a practical hybrid policy with measurable benefit.

## 9) Discussion Questions (Need Your Decision)

To start implementation efficiently, I need your choices on these:

1. Primary target for the next sprint:
   - HotpotQA optimization,
   - StrategyQA stability,
   - LogicQA recovery,
   - hybrid router first.

2. Compute policy:
   - maximize accuracy regardless of cost,
   - strict budget parity,
   - both tracked in parallel.

3. Paper positioning:
   - "A* as method improvement",
   - "A* + CoT complementary hybrid",
   - "budget-aware reasoning strategy".

4. Preferred evaluation scale for immediate runs:
   - fast pilot (100-200 samples),
   - medium (500),
   - full where available.

Once you pick these four, I will begin direct implementation in scripts and generate the first result table.
