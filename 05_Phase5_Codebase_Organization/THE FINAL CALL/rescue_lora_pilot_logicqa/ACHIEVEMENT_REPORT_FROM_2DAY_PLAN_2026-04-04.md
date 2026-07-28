# A to CoT Rescue LoRA Report (from 2-day plan)

Date: 2026-04-04
Project: rescue_lora_pilot_logicqa
Reference plan: THE FINAL CALL/astar_to_cot_rescue_lora_2day_plan.pdf

## 1) Executive Summary
This report documents what was completed from the 2-day rescue distillation plan, what results were achieved, where the fine-tuned models are saved, and what remains incomplete.

Core outcome:
- The rescue-distillation hypothesis showed positive signal.
- Best run so far is Fix-3.
- Best adapter metrics on this dev slice:
  - overall_accuracy = 28.57
  - rescue_rate = 33.33
  - preservation_rate = 25.00

## 2) Planned Objective (from PDF)
Primary objective from the plan:
- Repair a subset of CoT failures by distilling successful A-style traces into short linear rationales.
- Preserve existing CoT-correct behavior while improving rescue on CoT-wrong/A-correct cases.

Primary metrics from the plan:
- Overall accuracy
- Rescue rate on old A-only cases
- Preservation rate on old CoT-correct cases
- Category-wise repair view

## 3) Scope Actually Used
Planned first choice was StrategyQA, but this workspace had poor CoT/A pairing for that 100-sample split.

Applied fallback:
- Dataset used: LogicQA (paired overlap available and stable)
- Base model used: Qwen/Qwen2.5-0.5B-Instruct
- Method: LoRA adapter training (PEFT)

## 4) What Was Done
### Day 1 (Data pipeline)
Completed:
- Paired CoT/A output alignment and bucket construction
- Rescue rationale rewriting
- Hard filtering
- LoRA train/dev JSONL creation

Produced artifacts:
- data/rescue_pairs/summary.json
- data/rescue_pairs/rescue_manifest.csv
- data/rescue_pairs/rewritten_rescues.jsonl
- data/rescue_pairs/rewritten_rescues.filtered.jsonl
- data/rescue_pairs/filter_stats.json
- data/lora_train/train.jsonl
- data/lora_train/dev.jsonl
- data/eval/dev_eval_questions.jsonl
- data/eval/dev_bucket_labels.csv

### Day 2 (Training and evaluation)
Completed:
- Baseline eval (no adapter)
- Run-1 LoRA training + eval
- Fix-1 LoRA training + eval
- Additional continuation runs Fix-2 and Fix-3
- Live command logging in day2 log

Produced artifacts:
- outputs/adapters/.../final (multiple adapters)
- outputs/metrics/rescue_eval_qwen05b*.json
- outputs/metrics/sample_outputs_qwen05b*.csv

## 5) Data Construction Results
From data/rescue_pairs/summary.json:
- total_overlap = 100
- bucket_counts:
  - both_correct = 20
  - cot_only = 19
  - astar_only (rescue bucket) = 13
  - both_wrong = 48
- dev_bucket_counts:
  - astar_only = 3
  - cot_only = 4
  - both_wrong = 10
  - both_correct = 4

From data/rescue_pairs/filter_stats.json:
- rewritten input = 13
- kept = 13
- rejected = 0
- kept_rate = 100%

## 6) Training Runs and Metrics
Evaluation split size: total_dev = 21

| Run | Adapter | Overall | Rescue | Preservation |
|---|---|---:|---:|---:|
| Baseline | no adapter | 4.76 | 0.00 | 0.00 |
| Run-1 | rescue_lora_logicqa_qwen05b | 19.05 | 33.33 | 0.00 |
| Fix-1 | rescue_lora_logicqa_qwen05b_fix1 | 14.29 | 33.33 | 12.50 |
| Fix-2 | rescue_lora_logicqa_qwen05b_fix2 | 4.76 | 0.00 | 0.00 |
| Fix-3 | rescue_lora_logicqa_qwen05b_fix3 | 28.57 | 33.33 | 25.00 |

Category-wise repair observed in rescue bucket:
- logical_constraints: 1/3 repaired in Run-1, Fix-1, Fix-3

## 7) Findings
1. Rescue signal is real.
- Rescue moved from 0.00 (baseline) to 33.33 in the successful adapter runs.

2. Preservation is the critical bottleneck.
- Early runs showed poor preservation.
- Fix-3 improved preservation to 25.00 while keeping rescue at 33.33.

3. Controlled changes matter.
- Fix-2 (heavy preservation weighting + shorter training) collapsed to baseline behavior.
- Fix-3 (controlled epoch increase with stronger Fix-1 recipe) gave the best combined outcome.

4. Current best adapter for this pilot is Fix-3.

## 8) Plan Compliance Status
Completed:
- One dataset, one base model pilot implemented end to end
- Rescue + preserve style training set construction
- Rationale rewrite + hard filtering
- LoRA adapter training and held-out dev evaluation
- All core metrics from plan computed

Partially completed:
- Non-negotiable logs were initially backfilled from artifacts; now live day2 entries are present
- Manual qualitative review checklist items are not fully documented

Not fully aligned with strict PDF wording:
- StrategyQA first choice was not used (fallback to LogicQA due pairing issue)
- Manifest currently does not include explicit rewrite_status and verification_status columns
- Verification uses deterministic hard filters; separate explicit YES/NO verifier stage is not fully wired in the pipeline

## 9) Where Is the Fine-tuned Model?
Best fine-tuned adapter (current best):
- Relative path:
  - outputs/adapters/rescue_lora_logicqa_qwen05b_fix3/final
- Absolute path:
  - /home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/outputs/adapters/rescue_lora_logicqa_qwen05b_fix3/final
- Main weights file:
  - adapter_model.safetensors

Other trained adapters:
- /home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/outputs/adapters/rescue_lora_logicqa_qwen05b/final
- /home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/outputs/adapters/rescue_lora_logicqa_qwen05b_fix1/final
- /home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/outputs/adapters/rescue_lora_logicqa_qwen05b_fix2/final
- /home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/outputs/adapters/rescue_lora_logicqa/final (tiny smoke run)

## 10) Key Files for Export/Appendix
- notes/day1_log.md
- notes/day2_log.md
- outputs/metrics/rescue_eval_qwen05b_baseprompt.json
- outputs/metrics/rescue_eval_qwen05b.json
- outputs/metrics/rescue_eval_qwen05b_fix1.json
- outputs/metrics/rescue_eval_qwen05b_fix2.json
- outputs/metrics/rescue_eval_qwen05b_fix3.json
- outputs/metrics/sample_outputs_qwen05b_fix3.csv
- PILOT_RUN_REPORT_2026-04-04.md

## 11) Bottom Line
The pilot objective was achieved at proof-of-signal level.
- A subset of CoT failures is recoverable through rescue distillation.
- Best observed tradeoff in this workspace is currently the Fix-3 adapter.
- Preservation remains below desired level for production-grade use, so one more focused preservation-improvement round is recommended before final publication-level claims.
