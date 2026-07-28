#!/bin/bash
# Fair comparison run — both CoT and A* use same VLLM server + same evaluator
MODEL="meta-llama/Llama-3.1-8B-Instruct"
BASE_URL="http://localhost:8000/v1"
OUT="/home/dibyanayan/tanzeel/1ST_MAY_CONSISTENT"
LIMIT=2290

mkdir -p "$OUT/cot_results" "$OUT/astar_results"

echo "=== HotpotQA CoT (Fair Evaluator) ===" && date
conda run -n astar python "$OUT/hotpotqa_cot.py" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "none" \
    --limit $LIMIT \
    --output "$OUT/cot_results/hotpotqa_cot.json" \
    2>&1 | tee "$OUT/cot_results/run.log"

echo "=== HotpotQA A* (Fair Evaluator) ===" && date
conda run -n astar python "$OUT/hotpotqa_astar.py" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "none" \
    --limit $LIMIT \
    --no-shuffle \
    --output "$OUT/astar_results/hotpotqa_astar.json" \
    2>&1 | tee "$OUT/astar_results/run.log"

echo "=== All done ===" && date
