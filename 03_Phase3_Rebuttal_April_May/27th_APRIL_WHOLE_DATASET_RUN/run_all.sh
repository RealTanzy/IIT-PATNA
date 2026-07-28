#!/bin/bash
# ============================================================
# 27th April — Full Dataset Run
# Model: llama3.1:8b via Ollama
# GPU: CUDA_VISIBLE_DEVICES=0
# Datasets:
#   StrategyQA : 2290 (full test set)
#   HotpotQA   : 2290 (from validation split)
#   LogicQA    : SKIPPED (results already exist)
# Total: 4 runs (2 datasets × A* + CoT)
# ============================================================

export CUDA_VISIBLE_DEVICES=0

BASE=/home/dibyanayan/tanzeel
OUT=$BASE/27th_APRIL_WHOLE_DATASET_RUN
MODEL="llama3.1:8b"

echo "============================================================"
echo "27th April Full Dataset Run — START: $(date)"
echo "Model: $MODEL | GPU: CUDA_VISIBLE_DEVICES=0"
echo "StrategyQA=2290, HotpotQA=2290 | LogicQA=SKIPPED"
echo "============================================================"

# ----------------------------------------------------------
# RUN 1: StrategyQA CoT (2290 samples — full test set)
# ----------------------------------------------------------
echo ""
echo "[1/4] StrategyQA CoT — $(date)"
python "$BASE/COT Reasoning qwen/strategyqa_cot.py" \
    --model "$MODEL" \
    --limit 2290 \
    --output "$OUT/strategyqa/cot/results.json" \
    2>&1 | tee "$OUT/strategyqa/cot/run.log"
echo "[1/4] StrategyQA CoT DONE — $(date)"

# ----------------------------------------------------------
# RUN 2: StrategyQA A* (2290 samples — full test set)
# ----------------------------------------------------------
echo ""
echo "[2/4] StrategyQA A* — $(date)"
python "$BASE/FINAL_ASTAR/strategyqa_astar.py" \
    --model "$MODEL" \
    --limit 2290 \
    --output "$OUT/strategyqa/astar/results.json" \
    2>&1 | tee "$OUT/strategyqa/astar/run.log"
echo "[2/4] StrategyQA A* DONE — $(date)"

# ----------------------------------------------------------
# RUN 3: HotpotQA CoT (2290 samples)
# ----------------------------------------------------------
echo ""
echo "[3/4] HotpotQA CoT — $(date)"
python "$BASE/COT Reasoning qwen/hotpotqa_cot.py" \
    --model "$MODEL" \
    --limit 2290 \
    --output "$OUT/hotpotqa/cot/results.json" \
    2>&1 | tee "$OUT/hotpotqa/cot/run.log"
echo "[3/4] HotpotQA CoT DONE — $(date)"

# ----------------------------------------------------------
# RUN 4: HotpotQA A* (2290 samples)
# ----------------------------------------------------------
echo ""
echo "[4/4] HotpotQA A* — $(date)"
python "$BASE/FINAL_ASTAR/hotpotqa_astar.py" \
    --model "$MODEL" \
    --limit 2290 \
    --output "$OUT/hotpotqa/astar/results.json" \
    2>&1 | tee "$OUT/hotpotqa/astar/run.log"
echo "[4/4] HotpotQA A* DONE — $(date)"

echo ""
echo "============================================================"
echo "ALL 4 RUNS COMPLETE — $(date)"
echo "Results saved in: $OUT"
echo "============================================================"
