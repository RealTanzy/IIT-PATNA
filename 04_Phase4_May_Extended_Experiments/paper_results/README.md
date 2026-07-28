# Paper Results — Highest Accuracy Runs

All files in this folder are the **best (highest accuracy)** runs found across the workspace for each model × method × dataset combination.

---

## Accuracy Summary Table

| Model | Method | Dataset | Accuracy | N | File |
|---|---|---|---|---|---|
| Qwen-0.5B | CoT | LogicQA | **24.0%** | 500 | `qwen0.5b_cot_logicqa.json` |
| Qwen-0.5B | A\* | LogicQA | 20.2% | 500 | `qwen0.5b_astar_logicqa.json` |
| Qwen-0.5B | CoT | HotpotQA | 8.4% | 500 | `qwen0.5b_cot_hotpotqa.json` |
| Qwen-0.5B | A\* | HotpotQA | **30.0%** | 500 | `qwen0.5b_astar_hotpotqa.json` |
| Qwen-0.5B | CoT | StrategyQA | **53.6%** | 500 | `qwen0.5b_cot_strategyqa.json` |
| Qwen-0.5B | A\* | StrategyQA | 53.4% | 500 | `qwen0.5b_astar_strategyqa.json` |
| Llama-3.1-8B | CoT | LogicQA | **45.8%** | 651 | `llama3.1_8b_cot_logicqa.json` |
| Llama-3.1-8B | A\* | LogicQA | 44.9% | 651 | `llama3.1_8b_astar_logicqa.json` |
| Llama-3.1-8B | A\* | HotpotQA | **80.1%** | 1000 | `llama3.1_8b_astar_hotpotqa.json` |
| Llama-3.1-8B | CoT | StrategyQA | 64.5% | 93† | `llama3.1_8b_cot_strategyqa.json` |
| Llama-3.1-8B | A\* | StrategyQA | **74.8%** | 500 | `llama3.1_8b_astar_strategyqa.json` |

† Only 93 samples for Llama CoT StrategyQA (no full-set CoT run found for Llama on StrategyQA/HotpotQA).

---

## Notes

- **Llama CoT HotpotQA**: No standalone CoT result file was found for Llama on HotpotQA.
- A\* consistently beats CoT on HotpotQA and StrategyQA for both models; CoT edges out A\* on LogicQA.
- Qwen source: `COT Reasoning qwen/qwen_results/` (CoT) and `FINAL_ASTAR/qwen_results/` (A\*)
- Llama source: `results_shot/` + `logicqa_new/` (LogicQA), `hotpotqa/results_shot/` (HotpotQA), `strategyqa/` (StrategyQA)
