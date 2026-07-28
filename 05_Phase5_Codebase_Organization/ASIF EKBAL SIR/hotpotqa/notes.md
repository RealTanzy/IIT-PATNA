# HOTPOTQA - 100 Sample Comparison Notes

## Metrics

| Method | Accuracy | Correct/Total |
|---|---:|---:|
| Baseline A* | 78.0% | 78/100 |
| Baseline CoT | 53.0% | 53/100 |
| Previous Approach (deterministic rewrite + synthesis) | 39.0% | 39/100 |
| New Approach (vLLM rewrite + synthesis) | 26.0% | 26/100 |

## Deltas

- New vs Baseline A*: -52.0 pp
- New vs Previous: -13.0 pp
- Previous vs Baseline A*: -39.0 pp

## Source Files

- Baseline A*: `Tanzeel_astar/Dibyanayan_results/hotpotqa_astar_100_llama3.1_8b_device0.json`
- Baseline CoT: `Tanzeel_astar/Dibyanayan_results/hotpotqa_cot_100_llama3.1_8b_device0.json`
- Previous Approach: `PHASE_3/runs_100_deterministic_rewrite/hotpotqa/phase3_hotpotqa_100_det_rewrite_llama_synth.json`
- New Approach: `PHASE_3/runs_100_vllm_rewrite/hotpotqa/phase3_hotpotqa_100_vllm_rewrite_llama_synth.json`