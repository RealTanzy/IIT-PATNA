# Llama 3.1 8B Multi-Rerun Accuracy Results

Date: 2026-04-05
Folder: THE FINAL CALL/llama 3.1 8b

## Goal
Run multiple reruns and compare accuracy on the same dev split.

## Fixed Data Setup
- train records: 47
- dev records: 11
- eval split size: 21
- data recipe: preserve-multiplier=4.0, answer-only-ratio=0.15, seed=42

## Rerun Configs
1. fix1 (existing best before reruns)
- epochs=1.3, lr=5e-5

2. rerun1
- epochs=1.0, lr=5e-5

3. rerun2
- epochs=1.6, lr=5e-5

4. rerun3
- epochs=1.3, lr=3e-5

## Metrics Comparison
| Run | Overall | Rescue | Preservation |
|---|---:|---:|---:|
| fix1 | 33.33 | 33.33 | 50.00 |
| rerun1 | 14.29 | 0.00 | 12.50 |
| rerun2 | 14.29 | 0.00 | 12.50 |
| rerun3 | 19.05 | 0.00 | 25.00 |

## Best Run
Best by overall accuracy:
- fix1: overall 33.33, rescue 33.33, preservation 50.00

## Conclusion
- Yes, multiple reruns were executed and compared.
- In this sweep, none of the new reruns beat the existing fix1 run.
- Keep fix1 adapter as current best Llama 3.1 8B checkpoint.

## Best Model Path
- /home/dibyanayan/tanzeel/THE FINAL CALL/llama 3.1 8b/outputs/adapters/rescue_lora_llama31_8b_fix1/final

## Key Metrics Files
- outputs/metrics/rescue_eval_llama31_8b_fix1.json
- outputs/metrics/rescue_eval_llama31_8b_rerun1.json
- outputs/metrics/rescue_eval_llama31_8b_rerun2.json
- outputs/metrics/rescue_eval_llama31_8b_rerun3.json
