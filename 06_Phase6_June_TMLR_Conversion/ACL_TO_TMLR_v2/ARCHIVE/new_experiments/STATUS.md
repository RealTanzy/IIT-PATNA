# New Experiments - Status

## COMPLETE

### 8b_cot_astar_sc_tot/ (LLaMA-3.1-8B, Code Benchmarks)
| File | Dataset | Methods | Accuracy |
|------|---------|---------|----------|
| humaneval_results.json | HumanEval | CoT: 67.5%, A*: 64.4% | ✓ |
| humaneval_sc_results.json | HumanEval | SC: 65.0% | ✓ |
| humaneval_tot_results.json | HumanEval | ToT: 71.8% | ✓ |
| mbpp_results.json | MBPP | CoT: 39.8%, A*: 36.8% | ✓ |
| mbpp_sc_results.json | MBPP | SC: 38.0% | ✓ |
| mbpp_tot_results.json | MBPP | ToT: 40.4% | ✓ |
| codecontests_results.json | CodeContests | CoT: 100%, A*: 87% | ✓ |
| codecontests_sc_results.json | CodeContests | SC: 99.0% | ✓ |
| codecontests_tot_results.json | CodeContests | ToT: 91.2% | ✓ |

### 14b_cot_astar_sc_tot/ (Qwen2.5-14B, Code Benchmarks)
| File | Dataset | Methods | Accuracy |
|------|---------|---------|----------|
| humaneval_results.json | HumanEval | CoT+A*: 76.9% | ✓ |
| humaneval_sc_results.json | HumanEval | SC: 76.1% | ✓ |
| humaneval_tot_results.json | HumanEval | ToT: 79.8% | ✓ |
| mbpp_results.json | MBPP | CoT+A*: 45.8% | ✓ |
| mbpp_sc_results.json | MBPP | SC: 46.8% | ✓ |
| mbpp_tot_results.json | MBPP | ToT: 49.8% | ✓ |
| codecontests_results.json | CodeContests | CoT+A*: 97.0% | ✓ |
| codecontests_sc_results.json | CodeContests | SC: 98.2% | ✓ |
| codecontests_tot_results.json | CodeContests | ToT: 94.0% | ✓ |

### 05b_cot_astar_sc_tot/ (Qwen2.5-0.5B, Code Benchmarks)
| File | Dataset | Methods | Accuracy |
|------|---------|---------|----------|
| humaneval_results.json | HumanEval | CoT: 33.7%, A*: 20.2% | ✓ |
| humaneval_sc_results.json | HumanEval | SC: 26.4% | ✓ |
| humaneval_tot_results.json | HumanEval | ToT: 33.1% | ✓ |
| mbpp_results.json | MBPP | CoT: 24.0%, A*: 20.8% | ✓ |
| mbpp_sc_results.json | MBPP | SC: 21.4% | ✓ |
| mbpp_tot_results.json | MBPP | ToT: 23.0% | ✓ |
| codecontests_results.json | CodeContests | CoT: 49.8%, A*: 68.2% | ✓ |
| codecontests_sc_results.json | CodeContests | SC: 48.8% | ✓ |
| codecontests_tot_results.json | CodeContests | ToT: 67.8% | ✓ |

### 14b_math_cot_astar/ (Qwen2.5-14B, Math Benchmarks)
| File | Dataset | Accuracy |
|------|---------|----------|
| gsm8k_results.json | GSM8K | CoT: 61.6%, A*: 80.6% | ✓ |
| math_results.json | MATH | CoT: 41.8%, A*: 45.3% | ✓ |

### 05b_math_cot_astar/ (Qwen2.5-0.5B, Math Benchmarks)
| File | Dataset | Accuracy |
|------|---------|----------|
| gsm8k_results.json | GSM8K | CoT: 26.8%, A*: 14.5% | ✓ |
| math_results.json | MATH | CoT: 32.8%, A*: 16.2% | ✓ |

### routers_8b/ (LLaMA-3.1-8B, Code Benchmarks)
All 12 files complete (4 routers x 3 datasets) ✓

### routers_14b/ (Qwen2.5-14B, Code + Math)
All 20 files complete (4 routers x 5 datasets: HumanEval, MBPP, CC, GSM8K, MATH) ✓

---

## ON THE WAY (Running on server via master script, ~3-4 hours remaining)

These are currently being generated. Once done, they will be downloaded into the respective folders.

### routers_05b/ — 0.5B GSM8K/MATH Routers (7 files missing)

Running `run_routers_math.py 05b` on server. Uses vLLM Qwen-0.5B for SE sampling, Ollama qwen2.5:14b for Critic/Judge.

| File | What it does | Model used |
|------|-------------|------------|
| gsm8k_semantic_entropy.json | 5 samples per problem, disagreement >= 3 → A* | Qwen-0.5B |
| gsm8k_llm_critic.json | Compares CoT vs A* traces, picks better | qwen2.5:14b judge |
| gsm8k_llm_judge.json | Same as critic but <winner>A/B</winner> format | qwen2.5:14b judge |
| math_random_forest.json | Hand-crafted features, RF classifier, 5-fold CV | No LLM |
| math_semantic_entropy.json | Same as gsm8k SE but on MATH 500 | Qwen-0.5B |
| math_llm_critic.json | Same as gsm8k critic but on MATH | qwen2.5:14b judge |
| math_llm_judge.json | Same as gsm8k judge but on MATH | qwen2.5:14b judge |

### QA LLM Judge (6 files missing, Step 5 of master script)

Running `run_qa_judge.py` after 0.5B math routers finish. Adds LLM Judge routing for QA datasets at both scales.

| File | Folder | Input data |
|------|--------|-----------|
| strategyqa_llm_judge.json | routers_14b/ | 14B CoT+A* from ~/tanzeel/14b_dataset_check/results_14b/strategyqa/ |
| hotpotqa_llm_judge.json | routers_14b/ | 14B CoT+A* from ~/tanzeel/14b_dataset_check/results_14b/hotpotqa/ |
| logicqa_llm_judge.json | routers_14b/ | 14B CoT+A* from ~/tanzeel/14b_dataset_check/results_14b/logicqa/ |
| strategyqa_llm_judge.json | routers_05b/ | 0.5B data from ~/tanzeel/router_results/qwen_0.5b/ |
| hotpotqa_llm_judge.json | routers_05b/ | 0.5B data from ~/tanzeel/router_results/qwen_0.5b/ |
| logicqa_llm_judge.json | routers_05b/ | 0.5B data from ~/tanzeel/router_results/qwen_0.5b/ |

---

## NEEDS COPYING FROM SERVER (existing data, already generated previously, no compute needed)

QA routers for StrategyQA, HotpotQA, LogicQA already exist on server at:
- `~/tanzeel/router_results/qwen_14b/` → copy into `routers_14b/`
- `~/tanzeel/router_results/qwen_0.5b/` → copy into `routers_05b/`

Files to copy (12 per scale = 24 total):
- `{dataset}_semantic_entropy.json`
- `{dataset}_random_forest.json`
- `{dataset}_topology.json`
- `{dataset}_llm_critic.json`

For datasets: strategyqa, hotpotqa, logicqa
