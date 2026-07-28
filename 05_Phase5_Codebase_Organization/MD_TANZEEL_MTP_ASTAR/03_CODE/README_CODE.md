# Code — Reference Guide

All scripts here ran the actual experiments. They use Ollama for local
LLM inference (no API keys needed, all models are open-weight).

---

## A* Search Implementations (`astar_implementations/`)

### hotpotqa_astar.py
Runs A* search on HotpotQA multi-hop questions.
- **Model:** Any Ollama model (default: llama3.1:8b)
- **How to run:**
  ```bash
  python hotpotqa_astar.py --limit 500 --model llama3.1:8b
  ```
- **Output:** JSON file with per-question results (question, gold, pred, correct, f1, nodes, trace)
- **Expected runtime:** ~6-8 hours for 500Q on CPU

### logicqa_astar.py
Runs A* search on LogicQA MCQ questions.
- **Model:** Any Ollama model
- **How to run:**
  ```bash
  python logicqa_astar.py --limit 651 --model llama3.1:8b
  ```

### strategyqa_astar.py
Runs A* search on StrategyQA yes/no questions.
- **Model:** Any Ollama model
- **How to run:**
  ```bash
  python strategyqa_astar.py --limit 500 --model llama3.1:8b
  ```

---

## Chain-of-Thought Implementations (`cot_implementations/`)

These scripts match the Qwen-0.5B CoT baselines used in the paper.

### hotpotqa_cot.py
- Runs CoT prompting on HotpotQA
- **How to run:** `python hotpotqa_cot.py --limit 500 --model qwen:0.5b`

### logicqa_cot.py
- Runs CoT prompting on LogicQA
- **How to run:** `python logicqa_cot.py --limit 651 --model qwen:0.5b`

### strategyqa_cot.py
- Runs CoT prompting on StrategyQA
- **How to run:** `python strategyqa_cot.py --limit 500 --model qwen:0.5b`

---

## Qwen Matched Re-run (`qwen_matched_rerun/`)

### run_qwen_matched_cot.py
The script that produced the matched 500Q Qwen-0.5B CoT results
(hotpotqa_cot_500q.json). Runs CoT on the exact same 500 questions used in
the A* experiment. Supports resume (saves every 10 questions).

- **How to run:**
  ```bash
  ollama pull qwen:0.5b
  python run_qwen_matched_cot.py
  ```
- **Output:** qwen_matched_cot_results.json (already in 02_RESULTS/qwen_0.5B/)
- **Expected runtime:** ~45 minutes on CPU

---

## Phase 2 Heuristic Ablation (`phase2_ablation/`)

### phase2_strategyqa_astar_rewrite.py
Runs A* with 4 different heuristic configurations (BFS/depth/coverage/full)
on StrategyQA. Produces the ablation table in the paper (Section 6.2).

- **How to run:**
  ```bash
  python phase2_strategyqa_astar_rewrite.py --heuristic-mode astar --budget 15
  ```
- Valid `--heuristic-mode` values: `astar`, `bfs`, `greedy`

---

## Analysis Scripts (`analysis/`)

### compute_stats.py
Recomputes all confidence intervals (Wilson 95%) and McNemar p-values
for every comparison in the paper. Prints formatted tables.

```bash
# Run from MD_TANZEEL_MTP_ASTAR/ parent directory (tanzeel server backup/)
python TMLR_NEW_PAPER/compute_stats.py
```

### check_paper.py
Audits main.tex for: placeholders, missing citations, undefined references,
unmatched environments. Run after any LaTeX edits.

```bash
python check_paper.py
```

### analyze_results.py
Generates markdown tables from rebuttal experiment results. Used during the
May 2026 reviewer response period.

---

## Prerequisites

```bash
# Install Ollama: https://ollama.com
ollama pull qwen:0.5b         # 394 MB
ollama pull llama3.1:8b       # ~4.7 GB
ollama pull deepseek-r1:14b   # ~8 GB

# Python dependencies
pip install openai datasets tqdm requests
```

All scripts point to Ollama at http://localhost:11434 by default.
