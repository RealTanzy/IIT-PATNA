#!/bin/bash
# ============================================================
# Exp 3: Branching Factor Sensitivity — K ∈ {1, 2, 3, 4}
# Addresses: Reviewer 9TPm W5 (sensitivity analysis)
#
# Fixed: heuristic=astar, budget=30
# Varying: branch_k (candidates per expansion)
#
# K=1 is also the "linear chain" degenerate case (no branching)
# ============================================================

set -e
cd "$(dirname "$0")"

KS=(1 2 3 4)

if [[ ! -f samples/hotpotqa_300.json ]]; then
    echo "Sample files not found. Running prepare_samples.py first..."
    python3 prepare_samples.py
fi

echo "======================================================"
echo "  Experiment 3: Branching Factor Sensitivity"
echo "  K values: ${KS[*]}"
echo "  Dataset: HotpotQA (300 questions)"
echo "======================================================"

for K in "${KS[@]}"; do
    echo ""
    echo "--- HotpotQA | branch_k=$K ---"
    python3 hotpotqa_ablation.py \
        --heuristic-mode astar \
        --budget 30 \
        --branch-k "$K" \
        --out "results/hotpotqa_branchK${K}.json"
done

echo ""
echo "======================================================"
echo "  Experiment 3 complete. Generating table..."
echo "======================================================"
python3 analyze_results.py --exp branching
