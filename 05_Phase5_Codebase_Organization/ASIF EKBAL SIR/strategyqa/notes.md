# STRATEGYQA - 100 Sample Comparison Notes

## Metrics

| Method | Accuracy | Correct/Total |
|---|---:|---:|
| Baseline A* | 79.0% | 79/100 |
| Baseline CoT | 73.0% | 73/100 |
| Previous Approach (deterministic rewrite + synthesis) | 48.0% | 48/100 |
| New Approach (vLLM rewrite + synthesis) | 43.0% | 43/100 |

## Deltas

- New vs Baseline A*: -36.0 pp
- New vs Previous: -5.0 pp
- Previous vs Baseline A*: -31.0 pp

## Source Files

- Baseline A*: `Tanzeel_astar/Dibyanayan_results/strategyqa_astar_100_llama3.1_8b_device0.json`
- Baseline CoT: `Tanzeel_astar/Dibyanayan_results/strategyqa_cot_100_llama3.1_8b_device0.json`
- Previous Approach: `PHASE_3/runs_100_deterministic_rewrite/strategyqa/phase3_strategyqa_100_det_rewrite_llama_synth.json`
- New Approach: `PHASE_3/runs_100_vllm_rewrite/strategyqa/phase3_strategyqa_100_vllm_rewrite_llama_synth.json`