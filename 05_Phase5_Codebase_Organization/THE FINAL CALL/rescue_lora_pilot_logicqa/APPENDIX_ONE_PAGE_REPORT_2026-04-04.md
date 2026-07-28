# Appendix: 2-Day A to CoT Rescue LoRA Pilot (One-Page Summary)

Date: 2026-04-04  
Project: rescue_lora_pilot_logicqa  
Reference plan: THE FINAL CALL/astar_to_cot_rescue_lora_2day_plan.pdf

## A) Objective
Test whether some CoT failures can be repaired by distilling successful A-style traces into short linear rationales, then fine-tuning a lightweight LoRA adapter.

## B) Experimental Setup
- Dataset used: LogicQA (fallback from StrategyQA due pairing mismatch in available 100-sample files)
- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Method: LoRA (PEFT)
- Data recipe: preserve + rescue (+ small answer-only)
- Evaluation split: 21 dev examples

## C) Data Construction Snapshot
From paired CoT/A outputs (overlap = 100):
- both_correct: 20
- cot_only: 19
- astar_only (rescue): 13
- both_wrong: 48

Rescue rewrite/filter:
- input: 13
- kept: 13
- rejected: 0

## D) Core Results
| Run | Overall | Rescue Rate | Preservation Rate |
|---|---:|---:|---:|
| Baseline (no adapter) | 4.76 | 0.00 | 0.00 |
| Run-1 | 19.05 | 33.33 | 0.00 |
| Fix-1 | 14.29 | 33.33 | 12.50 |
| Fix-2 | 4.76 | 0.00 | 0.00 |
| Fix-3 | 28.57 | 33.33 | 25.00 |

Category-wise rescue repair observed:
- logical_constraints: 1/3 repaired in successful rescue runs

## E) Main Findings
1. Rescue hypothesis is supported at pilot scale.
- Adapter runs can recover part of the old CoT-wrong/A-correct bucket.

2. Preservation is the main risk variable.
- Some settings improved rescue but damaged preservation.

3. Controlled tuning helped.
- Fix-3 gave the best combined tradeoff on this dev slice.

## F) Current Best Fine-Tuned Model
Best adapter directory:
- outputs/adapters/rescue_lora_logicqa_qwen05b_fix3/final

Main adapter weights:
- outputs/adapters/rescue_lora_logicqa_qwen05b_fix3/final/adapter_model.safetensors

## G) Plan Compliance (Brief)
Completed:
- End-to-end pilot with data build, rewrite/filter, LoRA training, and metric evaluation.

Partially complete:
- Manual qualitative review steps are not fully documented.
- Manifest does not yet include explicit rewrite_status and verification_status fields.

## H) Conclusion
The 2-day pilot achieved proof-of-signal: a subset of CoT failures is recoverable through rescue distillation without search at inference time. The current best checkpoint is Fix-3, but preservation remains below production-level quality and should be improved in the next controlled iteration.
