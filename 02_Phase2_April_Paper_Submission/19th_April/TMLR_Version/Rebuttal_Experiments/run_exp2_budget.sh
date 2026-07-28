#!/bin/bash
# ============================================================
# Exp 2: Search Budget Sensitivity — B ∈ {5, 10, 15, 20, 30}
# Addresses: Reviewer 9TPm W5 (sensitivity analysis)
#
# Fixed: heuristic=astar, branch_k=3
# Varying: budget (max nodes to expand)
#
# Runs on HotpotQA only (most interesting, uses vLLM)
# For speed, uses the same 300-question sample as Exp 1
# ============================================================

set -e
cd "$(dirname "$0")"

BUDGETS=(5 10 15 20 30)

if [[ ! -f samples/hotpotqa_300.json ]]; then
    echo "Sample files not found. Running prepare_samples.py first..."
    python3 prepare_samples.py
fi

echo "======================================================"
echo "  Experiment 2: Budget Sensitivity"
echo "  Budgets: ${BUDGETS[*]}"
echo "  Dataset: HotpotQA (300 questions)"
echo "======================================================"

for B in "${BUDGETS[@]}"; do
    echo ""
    echo "--- HotpotQA | budget=$B ---"
    python3 hotpotqa_ablation.py \
        --heuristic-mode astar \
        --budget "$B" \
        --branch-k 3 \
        --out "results/hotpotqa_budget_B${B}.json"
done

echo ""
echo "======================================================"
echo "  Experiment 2 complete. Generating table..."
echo "======================================================"
python3 analyze_results.py --exp budget
