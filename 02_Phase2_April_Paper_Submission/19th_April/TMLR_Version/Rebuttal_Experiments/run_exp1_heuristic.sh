#!/bin/bash
# ============================================================
# Exp 1: Heuristic Ablation — BFS vs Greedy Best-First vs Full A*
# Addresses: Reviewer 9TPm W4, Reviewer tgwN W1/W2
#
# Tests: What does the heuristic actually contribute?
#   bfs    — h=0, degenerate to uniform-cost search (no critic)
#   greedy — f=h only (greedy best-first, critic score only)
#   astar  — f=g+h (standard A*, paper setting)
#
# Run: bash run_exp1_heuristic.sh [--hotpotqa-only]
# ============================================================

set -e
cd "$(dirname "$0")"

MODES=("bfs" "greedy" "astar")
DATASETS=("hotpotqa" "logicqa" "strategyqa")

HOTPOTQA_ONLY=false
[[ "$1" == "--hotpotqa-only" ]] && HOTPOTQA_ONLY=true

# ── Sanity: check sample files exist ──────────────────────────────────────────
if [[ ! -f samples/hotpotqa_300.json ]]; then
    echo "Sample files not found. Running prepare_samples.py first..."
    python3 prepare_samples.py
fi

echo "======================================================"
echo "  Experiment 1: Heuristic Ablation"
echo "  Modes: ${MODES[*]}"
echo "======================================================"

# ── HotpotQA (vLLM port 8000) ─────────────────────────────────────────────────
for MODE in "${MODES[@]}"; do
    echo ""
    echo "--- HotpotQA | heuristic=$MODE ---"
    python3 hotpotqa_ablation.py \
        --heuristic-mode "$MODE" \
        --budget 30 \
        --branch-k 3 \
        --out "results/hotpotqa_heuristic_${MODE}.json"
done

if $HOTPOTQA_ONLY; then
    echo "Skipping LogicQA and StrategyQA (--hotpotqa-only)"
    python3 analyze_results.py --exp heuristic
    exit 0
fi

# ── LogicQA (Ollama port 11434) ───────────────────────────────────────────────
for MODE in "${MODES[@]}"; do
    echo ""
    echo "--- LogicQA | heuristic=$MODE ---"
    python3 logicqa_ablation.py \
        --heuristic-mode "$MODE" \
        --budget 12 \
        --branch-k 2 \
        --out "results/logicqa_heuristic_${MODE}.json"
done

# ── StrategyQA (Ollama port 11434) ────────────────────────────────────────────
for MODE in "${MODES[@]}"; do
    echo ""
    echo "--- StrategyQA | heuristic=$MODE ---"
    python3 strategyqa_ablation.py \
        --heuristic-mode "$MODE" \
        --budget 15 \
        --branch-k 2 \
        --out "results/strategyqa_heuristic_${MODE}.json"
done

echo ""
echo "======================================================"
echo "  Experiment 1 complete. Generating table..."
echo "======================================================"
python3 analyze_results.py --exp heuristic
