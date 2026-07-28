# Rebuttal Experiments

Ablation experiments to address ACL ARR reviewer concerns.
All experiments use the **same fixed stratified sample** (prepared once by `prepare_samples.py`)
so every condition is evaluated on identical questions.

---

## Quick Start

```bash
cd 19th_April/TMLR_Version/Rebuttal_Experiments/

# 1. Prepare fixed sample sets (run ONCE)
python3 prepare_samples.py

# 2. Run all experiments
bash run_all_rebuttal.sh

# OR: Run only Exp 1 on HotpotQA (fastest, ~2h)
bash run_all_rebuttal.sh --exp1-only --hotpotqa-only
```

---

## Experiments

### Exp 1 — Heuristic Ablation (`run_exp1_heuristic.sh`)
**Addresses:** Reviewer 9TPm W4, Reviewer tgwN W1/W2

| Mode | Description |
|------|-------------|
| `bfs` | h=0 → A* degenerates to uniform-cost search (no critic at all) |
| `greedy` | f=h only → greedy best-first, critic drives everything |
| `astar` | f=g+h → standard A* (paper setting) |

Runs on all three datasets. Quantifies exactly what the DeepSeek critic heuristic contributes.

### Exp 2 — Budget Sensitivity (`run_exp2_budget.sh`)
**Addresses:** Reviewer 9TPm W5, Reviewer Gftu W3

Sweeps B ∈ {5, 10, 15, 20, 30} on HotpotQA. Shows compute–accuracy curve and confirms B=30 is near-optimal (diminishing returns).

### Exp 3 — Branching Factor Sensitivity (`run_exp3_branching.sh`)
**Addresses:** Reviewer 9TPm W5

Sweeps K ∈ {1, 2, 3, 4} on HotpotQA. K=1 is the degenerate linear-chain case — shows how much real branching contributes.

---

## Files

```
Rebuttal_Experiments/
├── prepare_samples.py          — create fixed stratified sample sets (run first)
├── hotpotqa_ablation.py        — HotpotQA unified ablation (vLLM port 8000)
├── logicqa_ablation.py         — LogicQA unified ablation (Ollama port 11434)
├── strategyqa_ablation.py      — StrategyQA unified ablation (Ollama port 11434)
├── analyze_results.py          — read results/, generate Markdown tables
├── run_exp1_heuristic.sh       — BFS vs Greedy vs A*
├── run_exp2_budget.sh          — budget sweep
├── run_exp3_branching.sh       — branching sweep
├── run_all_rebuttal.sh         — master runner
├── samples/                    — fixed sample JSONs (created by prepare_samples.py)
│   ├── hotpotqa_300.json
│   ├── logicqa_200.json
│   └── strategyqa_200.json
└── results/                    — output JSONs + Markdown tables go here
```

---

## Infrastructure

- **HotpotQA**: vLLM on `http://localhost:8000` serving `meta-llama/Llama-3.1-8B-Instruct`
- **LogicQA / StrategyQA**: Ollama on `http://localhost:11434` with `llama3.1:8b`
- **GPU**: NVIDIA A100 80GB (GPU 0 for vLLM, GPU 1 available for Ollama)

---

## Reviewer Mapping

| Experiment | Directly Addresses |
|---|---|
| Exp 1: BFS (h=0) vs Greedy vs A* | 9TPm W4 ("methodological comparison with BFS/DFS"), tgwN W1/W2 ("judge ablation", "only one heuristic") |
| Exp 2: Budget sweep | 9TPm W5 ("sensitivity analysis"), Gftu W3 ("compute–accuracy trade-off") |
| Exp 3: Branching sweep | 9TPm W5 ("sensitivity analysis") |

---

## Paper Baselines (full dataset, LLaMA 3.1-8B)

| Dataset | CoT | A* | Oracle |
|---|---|---|---|
| HotpotQA | 76.8% | 80.1% | 94.2% |
| LogicQA | 45.8% | 44.9% | — |
| StrategyQA | 64.5% | 74.8% | — |
