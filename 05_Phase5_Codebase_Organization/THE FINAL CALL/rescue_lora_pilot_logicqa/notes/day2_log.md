# Day 2 Log

## Log Provenance
- This log was backfilled from saved artifacts after runs completed.
- It is not a live process stream.
- At backfill time, no project process was running.
- Evidence files used:
	- outputs/metrics/train_qwen05b_fix1.log (2026-04-04 13:23:07 +0530)
	- outputs/metrics/eval_qwen05b_fix1.log (2026-04-04 13:25:05 +0530)
	- outputs/metrics/rescue_eval_qwen05b_fix1.json (2026-04-04 13:25:05 +0530)

## Goal (simple)
Train one LoRA adapter on the Day 1 data, evaluate rescue/preservation, and run at most one fix iteration.

## Checklist
- [x] Launch one LoRA run
- [x] Save adapter
- [x] Run evaluation
- [x] Compute rescue and preservation rates
- [ ] Inspect repaired/failure examples manually (not documented in this file)
- [x] Run one fix iteration (allowed by plan)

## Train Config
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA: r=16, alpha=16, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj

Run-1:
- epochs=1
- lr=1e-4
- batch_size=2
- grad_accum=4
- max_length=512
- train examples=33, dev examples=11
- adapter: `outputs/adapters/rescue_lora_logicqa_qwen05b/final`

Fix-1 (single small fix run):
- data rebalance: preserve multiplier 4.0, answer-only ratio 0.15
- epochs=1
- lr=5e-5
- batch_size=2
- grad_accum=4
- max_length=512
- train examples=47, dev examples=11
- adapter: `outputs/adapters/rescue_lora_logicqa_qwen05b_fix1/final`

## Eval Command
- Baseline (no adapter):
	- `python scripts/evaluate_rescue.py --base-model Qwen/Qwen2.5-0.5B-Instruct --out-json outputs/metrics/rescue_eval_qwen05b_baseprompt.json --out-csv outputs/metrics/sample_outputs_qwen05b_baseprompt.csv`
- Run-1 adapter:
	- `python scripts/evaluate_rescue.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path outputs/adapters/rescue_lora_logicqa_qwen05b/final --out-json outputs/metrics/rescue_eval_qwen05b.json --out-csv outputs/metrics/sample_outputs_qwen05b.csv`
- Fix-1 adapter:
	- `python scripts/evaluate_rescue.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path outputs/adapters/rescue_lora_logicqa_qwen05b_fix1/final --out-json outputs/metrics/rescue_eval_qwen05b_fix1.json --out-csv outputs/metrics/sample_outputs_qwen05b_fix1.csv`

## Final Metrics
- Baseline prompt-only:
	- overall_accuracy=4.76
	- rescue_rate=0.00 (0/3)
	- preservation_rate=0.00 (0/8)

- Run-1 adapter:
	- overall_accuracy=19.05
	- rescue_rate=33.33 (1/3)
	- preservation_rate=0.00 (0/8)

- Fix-1 adapter:
	- overall_accuracy=14.29
	- rescue_rate=33.33 (1/3)
	- preservation_rate=12.50 (1/8)

- Fix-2 adapter (preservation-focused trial):
	- overall_accuracy=4.76
	- rescue_rate=0.00 (0/3)
	- preservation_rate=0.00 (0/8)

- Fix-3 adapter (controlled change: epochs 1.3):
	- overall_accuracy=28.57
	- rescue_rate=33.33 (1/3)
	- preservation_rate=25.00 (2/8)

- Category-wise repair on rescue bucket:
	- logical_constraints repaired=1/3 (33.33)

## Notes for Next Iteration
- Positive: rescue signal appeared (0.00 -> 33.33).
- Main bottleneck: preservation is still low.
- Next practical direction: improve preserve supervision quality and answer extraction robustness, while protecting rescue gains.
- Fix-2 regressed to baseline-level behavior; keep Fix-1 as current best adapter.
- Fix-3 outperformed all prior runs on this dev slice; promote Fix-3 as current best adapter.

### Live Entry: 2026-04-04 14:46:14 +0530
- Command: python scripts/build_rescue_set.py --preserve-multiplier 10.0 --answer-only-ratio 0.25 --seed 42
- Run log: outputs/metrics/live_day2_20260404_144614.log
- End time: 2026-04-04 14:46:14 +0530
- Exit code: 0
- Status: success

### Live Entry: 2026-04-04 14:46:14 +0530
- Command: python scripts/make_lora_jsonl.py
- Run log: outputs/metrics/live_day2_20260404_144614.log
- End time: 2026-04-04 14:46:14 +0530
- Exit code: 0
- Status: success

### Live Entry: 2026-04-04 14:46:26 +0530
- Command: env CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py --base-model Qwen/Qwen2.5-0.5B-Instruct --output-dir outputs/adapters/rescue_lora_logicqa_qwen05b_fix2 --epochs 0.8 --lr 3e-5 --batch-size 2 --grad-accum 4 --max-length 512 --logging-steps 5
- Run log: outputs/metrics/live_day2_20260404_144626.log
- End time: 2026-04-04 14:46:43 +0530
- Exit code: 0
- Status: success

### Live Entry: 2026-04-04 14:46:47 +0530
- Command: env CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_rescue.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path outputs/adapters/rescue_lora_logicqa_qwen05b_fix2/final --out-json outputs/metrics/rescue_eval_qwen05b_fix2.json --out-csv outputs/metrics/sample_outputs_qwen05b_fix2.csv
- Run log: outputs/metrics/live_day2_20260404_144647.log
- End time: 2026-04-04 14:48:45 +0530
- Exit code: 0
- Status: success

### Live Entry: 2026-04-04 16:21:03 +0530
- Command: python scripts/build_rescue_set.py --preserve-multiplier 4.0 --answer-only-ratio 0.15 --seed 42
- Run log: outputs/metrics/live_day2_20260404_162103.log
- End time: 2026-04-04 16:21:03 +0530
- Exit code: 0
- Status: success

### Live Entry: 2026-04-04 16:21:03 +0530
- Command: python scripts/make_lora_jsonl.py
- Run log: outputs/metrics/live_day2_20260404_162103.log
- End time: 2026-04-04 16:21:03 +0530
- Exit code: 0
- Status: success

### Live Entry: 2026-04-04 16:21:08 +0530
- Command: env CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py --base-model Qwen/Qwen2.5-0.5B-Instruct --output-dir outputs/adapters/rescue_lora_logicqa_qwen05b_fix3 --epochs 1.3 --lr 5e-5 --batch-size 2 --grad-accum 4 --max-length 512 --logging-steps 5
- Run log: outputs/metrics/live_day2_20260404_162108.log
- End time: 2026-04-04 16:21:28 +0530
- Exit code: 0
- Status: success

### Live Entry: 2026-04-04 16:21:33 +0530
- Command: env CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_rescue.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path outputs/adapters/rescue_lora_logicqa_qwen05b_fix3/final --out-json outputs/metrics/rescue_eval_qwen05b_fix3.json --out-csv outputs/metrics/sample_outputs_qwen05b_fix3.csv
- Run log: outputs/metrics/live_day2_20260404_162133.log
- End time: 2026-04-04 16:23:43 +0530
- Exit code: 0
- Status: success
