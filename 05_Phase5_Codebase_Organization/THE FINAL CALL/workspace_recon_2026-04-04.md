# Workspace Reconnaissance (April 4, 2026)

## Scope
This note summarizes what has already been done across the TANZEEL workspace.
It is built from direct file/code reading plus JSON-level metric extraction.

## 1) Project Evolution (high-level)
- Early stage (`archive/`, early `hotpotqa/` and `strategyqa/`): foundational A* and CoT prototypes.
- Baseline scaling stage (`FINAL_ASTAR/`, `COT Reasoning qwen/`, `strategyqa/`, `logicqa_new/`, `hotpotqa/results_*`): clean A* vs CoT runs on Qwen-0.5B and Llama-3.1-8B.
- Performance deep-dive (`performance/`): disagreement-set reruns and majority-vote stability analysis.
- Phase 2 (`PHASE_2/`): explicit heuristic ablation with prompt-versioned search traces.
- Phase 3 (`PHASE_3/`): dual-trace pipeline (A* trace rewrite + CoT/A* synthesis), then multiple retries and variants.
- Structured comparison/reporting (`ASIF EKBAL SIR/`): dataset-wise metrics, reports, and dataset-aware ablations.

## 2) Best-verified baseline results (from files)

### Qwen-0.5B (500 examples)
- LogicQA: CoT 24.0, A* 20.2
- HotpotQA: CoT 8.4, A* 30.0
- StrategyQA: CoT 53.6, A* 53.4

### Llama-3.1-8B
- LogicQA (651): CoT 45.78, A* 44.85
- HotpotQA (1000): A* 80.1 (strongest preserved HotpotQA result)
- StrategyQA (500): A* 74.8
- StrategyQA CoT 64.5 exists on 93-item disagreement subset, not full 500.

### 100-sample Llama baselines (`Tanzeel_astar/Dibyanayan_results`)
- LogicQA: A* 33, CoT 39
- HotpotQA: A* 78, CoT 53
- StrategyQA: A* 79, CoT 73

## 3) Phase 2 findings (heuristic study)
- Code: `PHASE_2/phase2_strategyqa_astar_rewrite.py` with prompt library `PHASE_2/meta_prompts.py`.
- Main mechanism: heuristic mode in {none, depth, coverage, full}, with prompt-versioned traces.
- `PHASE_2/ANALYSIS_REPORT.md` highlights coverage heuristic as strongest in reported runs.
- `PHASE_2/results_continuous/` contains many pilot/full cycles; these files show run-to-run variation (not one single monotonic trend).

## 4) Phase 3 findings (dual-trace)
- Core idea: linearize/refine A* trace, then synthesize with CoT trace via LLM.
- Main runner: `PHASE_3/phase3_dual_trace_synthesis.py`.
- Includes answer-mode inference and dataset-specific normalization:
  - multiple_choice
  - yesno
  - freeform
- LogicQA 100-sample outcomes appear in multiple files:
  - 42 (main positive run)
  - 39 / 38 / 32 in alternate retries/variants
- This indicates Phase 3 is sensitive to configuration/infrastructure and not represented by a single stable point.

## 5) ASIF structured results and ablations (most organized folder)

### Canonical comparison notes/metrics
- `ASIF EKBAL SIR/hotpotqa/metrics.json`: baseline A* 78, baseline CoT 53, previous 39, new(vLLM rewrite) 26
- `ASIF EKBAL SIR/logicqa/metrics.json`: baseline A* 33, baseline CoT 39, previous 42, new 42
- `ASIF EKBAL SIR/strategyqa/metrics.json`: baseline A* 79, baseline CoT 73, previous 48, new 43

### Additional tracked runs
- `ASIF EKBAL SIR/experiments/proper_llm_device0/`:
  - LogicQA 37
  - HotpotQA 20
  - StrategyQA 25
- `ASIF EKBAL SIR/experiments/dataset_aware/` shows improved dataset-aware variants:
  - LogicQA around 40-41
  - HotpotQA around 48-53
  - StrategyQA around 62-64

Interpretation:
- Generic or improperly matched answer-space handling caused large regressions.
- Dataset-aware answer formatting/scoring is a key requirement for dual-trace approach.

## 6) Performance/disagreement analysis
- Folder: `performance/`
- Setup: 93 StrategyQA questions where original CoT and A* disagreed.
- 3 runs each for CoT and A*, then majority vote.
- Majority report indicates CoT majority > A* majority on that disagreement subset.
- Also shows high variance and a hard core where both methods fail.

## 7) Recovery experiment branch
- `astar_recovery/astar_recovery_experiment.py` reruns A* multiple times on A*-fail / CoT-win subset.
- Goal: separate systematic failures from sampling flukes.
- This branch is exploratory and complements `performance/` stability analysis.

## 8) Important contradictions to keep in mind
- Several markdown reports in `PHASE_3/` are not fully consistent with each other.
- JSON artifacts should be treated as primary evidence over prose claims.
- Some files are full object outputs (with top-level accuracy), others are raw arrays (accuracy must be computed from `correct` field).

## 9) Current best understanding for next work
- Baseline A* is very strong for HotpotQA (especially Llama).
- CoT is competitive or better on LogicQA.
- StrategyQA winner can depend on slice and protocol (full-run vs disagreement subset).
- Dual-trace synthesis is promising but unstable unless dataset-specific answer handling is enforced.
- For future experiments, strict split control and artifact-level metric auditing are mandatory.

## 10) Suggested source-of-truth hierarchy for future analysis
1. JSON result artifacts (computed metrics)
2. Dataset-level `metrics.json` files in `ASIF EKBAL SIR/`
3. Consolidated `paper_results/` snapshots
4. Markdown narrative reports (useful but secondary when conflicting)

