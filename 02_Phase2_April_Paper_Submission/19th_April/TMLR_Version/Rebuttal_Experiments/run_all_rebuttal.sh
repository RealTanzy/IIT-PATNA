#!/bin/bash
# ============================================================
# Master Runner — All Rebuttal Experiments
# Runs Exp 1 → 2 → 3 sequentially, then generates summary
#
# Usage:
#   bash run_all_rebuttal.sh                     # all experiments, all datasets
#   bash run_all_rebuttal.sh --hotpotqa-only      # faster: HotpotQA only
#   bash run_all_rebuttal.sh --exp1-only          # heuristic ablation only
#
# Estimated runtime (HotpotQA-only, 1 A100):
#   Exp 1: ~2h  (3 modes × 300 questions)
#   Exp 2: ~2h  (5 budgets × 300 questions)
#   Exp 3: ~2h  (4 K values × 300 questions)
#   Total: ~6h
# ============================================================

set -e
cd "$(dirname "$0")"

HOTPOTQA_ONLY=""
EXP1_ONLY=false

for arg in "$@"; do
    case $arg in
        --hotpotqa-only) HOTPOTQA_ONLY="--hotpotqa-only" ;;
        --exp1-only)     EXP1_ONLY=true ;;
    esac
done

# Step 0: prepare fixed sample sets (once)
if [[ ! -f samples/hotpotqa_300.json ]]; then
    echo ">>> Step 0: Preparing sample sets..."
    python3 prepare_samples.py
else
    echo ">>> Step 0: Sample sets already prepared — skipping."
fi

echo ""
echo ">>> Step 1: Heuristic Ablation (BFS / Greedy / A*)"
bash run_exp1_heuristic.sh $HOTPOTQA_ONLY

if ! $EXP1_ONLY; then
    echo ""
    echo ">>> Step 2: Budget Sensitivity (B = 5, 10, 15, 20, 30)"
    bash run_exp2_budget.sh

    echo ""
    echo ">>> Step 3: Branching Sensitivity (K = 1, 2, 3, 4)"
    bash run_exp3_branching.sh
fi

echo ""
echo ">>> Generating combined summary..."
python3 analyze_results.py --exp summary

echo ""
echo "======================================================"
echo "  All rebuttal experiments complete!"
echo "  Results: results/"
echo "  Summary: results/REBUTTAL_APPENDIX.md"
echo "======================================================"
