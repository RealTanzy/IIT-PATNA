# Day 1 Log

## Log Provenance
- This log was backfilled from saved artifacts after runs completed.
- It is not a live process stream.
- At backfill time, no project process was running.
- Evidence files used:
	- outputs/metrics/train_qwen05b.log (2026-04-04 13:19:17 +0530)
	- outputs/metrics/rescue_eval_qwen05b.json (2026-04-04 13:21:23 +0530)

## Goal (simple)
Build clean training data for "A* rescue distillation":
- find cases where CoT failed but A* was correct,
- rewrite those A* traces into short linear rationales,
- prepare train/dev JSONL for LoRA.

## Checklist
- [x] Export paired CoT/A* outputs
- [x] Build bucket manifest
- [x] Rewrite rescue traces
- [x] Filter rewrites
- [x] Build train/dev JSONL
- [ ] Inspect 30 samples manually (not documented in this log)

## Commands Run
- `python scripts/build_rescue_set.py`
- `python scripts/rewrite_rationales.py`
- `python scripts/filter_rationales.py`
- `python scripts/make_lora_jsonl.py`

## Counts
- Paired overlap: 100 examples.
- Buckets: both_correct=20, cot_only=19, astar_only=13, both_wrong=48.
- Train/dev bucket split:
	- train astar_only=10, dev astar_only=3
	- train cot_only=15, dev cot_only=4
	- train both_correct=16, dev both_correct=4
	- train both_wrong=38, dev both_wrong=10
- Initial run data size (run-1): train=33, dev=11.
- Rewrite/filter: input rescue=13, kept=13, rejected=0.

## Failure Modes Observed
- StrategyQA 100-sample files in this workspace were not cleanly paired for rescue bucketing.
- Needed to switch dataset for this pilot to maintain valid CoT/A* pairing.

## Decisions
- Used LogicQA for this 2-day pilot because overlap was clean and sufficient.
- Kept scope minimal: one dataset, one adapter run first, then evaluate.
- Used short rationale filtering (step/length/search-artifact checks) before training.

