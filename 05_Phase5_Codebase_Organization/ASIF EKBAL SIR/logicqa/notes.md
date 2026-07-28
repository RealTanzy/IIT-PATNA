# LOGICQA - 100 Sample Comparison Notes

## Metrics

| Method | Accuracy | Correct/Total |
|---|---:|---:|
| Baseline A* | 33.0% | 33/100 |
| Baseline CoT | 39.0% | 39/100 |
| Previous Approach (deterministic rewrite + synthesis) | 42.0% | 42/100 |
| New Approach (vLLM rewrite + synthesis) | 42.0% | 42/100 |

## Deltas

- New vs Baseline A*: +9.0 pp
- New vs Previous: +0.0 pp
- Previous vs Baseline A*: +9.0 pp

## Source Files

- Baseline A*: `Tanzeel_astar/Dibyanayan_results/logicqa_astar_100_llama3.1_8b_device0.json`
- Baseline CoT: `Tanzeel_astar/Dibyanayan_results/logicqa_cot_100_llama3.1_8b_device0.json`
- Previous Approach: `PHASE_3/logicqa_phase3_dualtrace_100_llama3.1_8b.json`
- New Approach: `PHASE_3/runs_100_vllm_rewrite/logicqa/phase3_logicqa_100_vllm_rewrite_llama_synth.json`