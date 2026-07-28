#!/bin/bash
OUT=/home/dibyanayan/tanzeel/27th_APRIL_WHOLE_DATASET_RUN
MODEL="meta-llama/Llama-3.1-8B-Instruct"
BASE_URL="http://localhost:8000/v1"
SCRIPTS="/home/dibyanayan/tanzeel/opus testing"

mkdir -p "$OUT/hotpotqa/cot" "$OUT/hotpotqa/astar"

echo "=== Starting HotpotQA CoT ===" && date
conda run -n astar python "$SCRIPTS/hotpotqa_cot_api.py" --model "$MODEL" --base-url "$BASE_URL" --api-key "none" --limit 2290 --output "$OUT/hotpotqa/cot/results.json" 2>&1 | tee "$OUT/hotpotqa/cot/run.log"

echo "=== Starting HotpotQA A* ===" && date
conda run -n astar python "$SCRIPTS/hotpotqa_astar_api.py" --model "$MODEL" --base-url "$BASE_URL" --api-key "none" --limit 2290 --output "$OUT/hotpotqa/astar/results.json" 2>&1 | tee "$OUT/hotpotqa/astar/run.log"

echo "=== All done ===" && date
