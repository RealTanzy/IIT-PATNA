#!/bin/bash
# Sequential runner — called AFTER prepare_samples.py has run.
# BFS is expected to already be running (or finished).
# This script runs all remaining HotpotQA ablation conditions.

set -e
BASE=/home/dibyanayan/tanzeel/19th_April/TMLR_Version/Rebuttal_Experiments
cd "$BASE"
mkdir -p logs results

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/master.log; }

# ─── Exp 1: greedy ───────────────────────────────────────────────────────────
log "=== Exp1: GREEDY ==="
python3 hotpotqa_ablation.py \
    --heuristic-mode greedy --budget 30 --branch-k 3 \
    --out results/hotpotqa_heuristic_greedy.json \
    2>&1 | tee logs/exp1_greedy.log
log "Greedy done."

# ─── Exp 1: full A* ──────────────────────────────────────────────────────────
log "=== Exp1: FULL ASTAR ==="
python3 hotpotqa_ablation.py \
    --heuristic-mode astar --budget 30 --branch-k 3 \
    --out results/hotpotqa_heuristic_astar.json \
    2>&1 | tee logs/exp1_astar.log
log "Full A* done."

# ─── Exp 2: budget sweep ─────────────────────────────────────────────────────
log "=== Exp2: BUDGET SWEEP ==="
for B in 5 10 15 20; do
    log "  Budget B=$B"
    python3 hotpotqa_ablation.py \
        --heuristic-mode astar --budget "$B" --branch-k 3 \
        --out "results/hotpotqa_budget_B${B}.json" \
        2>&1 | tee "logs/exp2_B${B}.log"
done
# B=30 reuses the astar result from Exp1
cp results/hotpotqa_heuristic_astar.json results/hotpotqa_budget_B30.json
log "Budget sweep done."

# ─── Exp 3: branching sweep ──────────────────────────────────────────────────
log "=== Exp3: BRANCHING SWEEP ==="
for K in 1 2 4; do
    log "  Branch K=$K"
    python3 hotpotqa_ablation.py \
        --heuristic-mode astar --budget 30 --branch-k "$K" \
        --out "results/hotpotqa_branchK${K}.json" \
        2>&1 | tee "logs/exp3_K${K}.log"
done
# K=3 reuses the astar result from Exp1
cp results/hotpotqa_heuristic_astar.json results/hotpotqa_branchK3.json
log "Branching sweep done."

# ─── Analysis ────────────────────────────────────────────────────────────────
log "=== Generating summary ==="
python3 analyze_results.py --exp all 2>&1 | tee logs/analysis.log
log "=== ALL DONE. Check results/REBUTTAL_APPENDIX.md ==="
