# Precision Rerun Report (Conda astar, GPU0, LLM rewrite + LLM synthesis)

All new reruns were executed with:
- Environment: `conda run -n astar`
- GPU: device 0
- Model for rewrite: `llama3.1:8b`
- Model for synthesis: `llama3.1:8b`
- Samples per dataset: 100

## Consolidated Metrics

| Dataset | Baseline A* | Baseline CoT | Previous | New (vLLM rewrite) | New (LLM rewrite, proper rerun) | LLM rows (proper rerun) | Fallback rows (proper rerun) |
|---|---:|---:|---:|---:|---:|---:|---:|
| logicqa | 33.0% | 39.0% | 42.0% | 42.0% | 37.0% | 100 | 0 |
| hotpotqa | 78.0% | 53.0% | 39.0% | 26.0% | 20.0% | 100 | 0 |
| strategyqa | 79.0% | 73.0% | 48.0% | 43.0% | 25.0% | 100 | 0 |

## Delta vs Baseline A* (proper rerun)

- logicqa: +4.0 pp
- hotpotqa: -58.0 pp
- strategyqa: -54.0 pp

## Source Files

### logicqa
- baseline_astar: `Tanzeel_astar/Dibyanayan_results/logicqa_astar_100_llama3.1_8b_device0.json`
- baseline_cot: `Tanzeel_astar/Dibyanayan_results/logicqa_cot_100_llama3.1_8b_device0.json`
- previous: `PHASE_3/logicqa_phase3_dualtrace_100_llama3.1_8b.json`
- new_vllm_rewrite: `PHASE_3/runs_100_vllm_rewrite/logicqa/phase3_logicqa_100_vllm_rewrite_llama_synth.json`
- new_llm_rewrite: `ASIF EKBAL SIR/experiments/proper_llm_device0/logicqa/phase3_logicqa_100_llm_rewrite_llm_synth_device0.json`

### hotpotqa
- baseline_astar: `Tanzeel_astar/Dibyanayan_results/hotpotqa_astar_100_llama3.1_8b_device0.json`
- baseline_cot: `Tanzeel_astar/Dibyanayan_results/hotpotqa_cot_100_llama3.1_8b_device0.json`
- previous: `PHASE_3/runs_100_deterministic_rewrite/hotpotqa/phase3_hotpotqa_100_det_rewrite_llama_synth.json`
- new_vllm_rewrite: `PHASE_3/runs_100_vllm_rewrite/hotpotqa/phase3_hotpotqa_100_vllm_rewrite_llama_synth.json`
- new_llm_rewrite: `ASIF EKBAL SIR/experiments/proper_llm_device0/hotpotqa/phase3_hotpotqa_100_llm_rewrite_llm_synth_device0.json`

### strategyqa
- baseline_astar: `Tanzeel_astar/Dibyanayan_results/strategyqa_astar_100_llama3.1_8b_device0.json`
- baseline_cot: `Tanzeel_astar/Dibyanayan_results/strategyqa_cot_100_llama3.1_8b_device0.json`
- previous: `PHASE_3/runs_100_deterministic_rewrite/strategyqa/phase3_strategyqa_100_det_rewrite_llama_synth.json`
- new_vllm_rewrite: `PHASE_3/runs_100_vllm_rewrite/strategyqa/phase3_strategyqa_100_vllm_rewrite_llama_synth.json`
- new_llm_rewrite: `ASIF EKBAL SIR/experiments/proper_llm_device0/strategyqa/phase3_strategyqa_100_llm_rewrite_llm_synth_device0.json`
