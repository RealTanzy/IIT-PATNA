# Pilot Run Report (April 4, 2026)

## 1) PDF Plan Read Status
The file was read and extracted successfully:
- `THE FINAL CALL/astar_to_cot_rescue_lora_2day_plan.pdf`
- Extracted text: `THE FINAL CALL/astar_to_cot_rescue_lora_2day_plan_extracted.txt`

The implementation below follows that plan (single dataset, rescue-focused, LoRA, 4-check eval, one small fix run only).

## 2) Dataset Choice
Requested first choice in the PDF is StrategyQA, but current 100-sample StrategyQA A*/CoT files are mostly unpaired in this workspace.

So the pilot used LogicQA where 100-sample CoT/A* records are cleanly paired.

## 3) Bucket Construction (LogicQA 100 overlap)
From `data/rescue_pairs/summary.json`:
- total overlap: 100
- buckets:
  - both_correct: 20
  - cot_only: 19
  - astar_only (rescue bucket): 13
  - both_wrong: 48

Train/dev split:
- train astar_only: 10
- dev astar_only: 3
- train preserve selected: 20 (run-1), 31 (fix-1)
- answer-only: 3 (run-1), 6 (fix-1)

## 4) Rescue Rewrite Stage
- input rescue examples: 13
- rewritten: 13/13
- filtered keep: 13/13

Artifacts:
- `data/rescue_pairs/rewritten_rescues.jsonl`
- `data/rescue_pairs/rewritten_rescues.filtered.jsonl`
- `data/rescue_pairs/filter_stats.json`

## 5) LoRA Training Runs
### Run-1
- base model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA target modules auto-detected:
  - q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj
- train examples: 33
- dev examples (supervised): 11
- epochs: 1
- lr: 1e-4

Adapter path:
- `outputs/adapters/rescue_lora_logicqa_qwen05b/final`

### Fix-1 (single allowed fix run)
Reason for fix: preservation was 0.0 in Run-1.

Changes:
- preserve multiplier increased to 4.0
- answer-only ratio increased to 0.15
- learning rate lowered to 5e-5
- all else same

Adapter path:
- `outputs/adapters/rescue_lora_logicqa_qwen05b_fix1/final`

## 6) Four Required Checks
Evaluated on held-out dev split (`n=21`) with bucket labels.

### Baseline prompt-only (no adapter)
- overall accuracy: 4.76
- rescue rate: 0.0 (0/3)
- preservation rate: 0.0 (0/8)

File:
- `outputs/metrics/rescue_eval_qwen05b_baseprompt.json`

### Run-1 adapter
- overall accuracy: 19.05
- rescue rate: 33.33 (1/3)
- preservation rate: 0.0 (0/8)
- category-wise repaired:
  - logical_constraints: 1/3

File:
- `outputs/metrics/rescue_eval_qwen05b.json`

### Fix-1 adapter (one tiny fix)
- overall accuracy: 14.29
- rescue rate: 33.33 (1/3)
- preservation rate: 12.5 (1/8)
- category-wise repaired:
  - logical_constraints: 1/3

File:
- `outputs/metrics/rescue_eval_qwen05b_fix1.json`

## 7) Interpretation
- Rescue signal is real in this pilot setup: adapter beats base prompt on rescue bucket (0.0 -> 33.33).
- Preservation is the current bottleneck.
- The single fix improved preservation (0.0 -> 12.5) but reduced overall accuracy vs Run-1.
- This matches the expected tradeoff in early rescue-distillation pilots.

## 8) Main Artifacts
- Train/dev data:
  - `data/lora_train/train.jsonl`
  - `data/lora_train/dev.jsonl`
- Eval questions:
  - `data/eval/dev_eval_questions.jsonl`
  - `data/eval/dev_bucket_labels.csv`
- Metrics:
  - `outputs/metrics/rescue_eval_qwen05b_baseprompt.json`
  - `outputs/metrics/rescue_eval_qwen05b.json`
  - `outputs/metrics/rescue_eval_qwen05b_fix1.json`
- Per-example outputs:
  - `outputs/metrics/sample_outputs_qwen05b.csv`
  - `outputs/metrics/sample_outputs_qwen05b_fix1.csv`

## 9) Immediate Next Step (for day-2 continuation)
Given this pilot, the most useful next single change is to improve preservation without sacrificing rescue by:
- adding more high-quality preserve examples (especially `cot_only` hard positives), and
- tightening output format extraction to reduce false negatives in preserve bucket.
