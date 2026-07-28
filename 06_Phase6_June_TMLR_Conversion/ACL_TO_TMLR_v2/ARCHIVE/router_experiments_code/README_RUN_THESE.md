# Router Experiments for Code Benchmarks

## What We Need
Run 4 routers (Semantic Entropy, LLM-as-Critic, LLM-as-Judge, Random Forest) on 3 code benchmarks (HumanEval, MBPP, CodeContests) using LLaMA-3.1-8B backbone.

## Expected Output Format
Each experiment should produce a JSON file with this structure:

```json
{
  "router": "semantic_entropy",
  "dataset": "humaneval", 
  "model": "llama-3.1-8b",
  "results": [
    {
      "task_id": "HumanEval/0",
      "cot_correct": true,
      "astar_correct": true,
      "router_choice": "cot",
      "router_correct": true
    },
    ...
  ],
  "summary": {
    "total": 163,
    "router_accuracy": 71.2,
    "cot_accuracy": 67.48,
    "astar_accuracy": 64.42,
    "oracle_accuracy": 74.85
  }
}
```

## File Naming
```
router_experiments_code/
├── humaneval_semantic_entropy.json
├── humaneval_llm_critic.json
├── humaneval_llm_judge.json
├── humaneval_random_forest.json
├── mbpp_semantic_entropy.json
├── mbpp_llm_critic.json
├── mbpp_llm_judge.json
├── mbpp_random_forest.json
├── codecontests_semantic_entropy.json
├── codecontests_llm_critic.json
├── codecontests_llm_judge.json
└── codecontests_random_forest.json
```
