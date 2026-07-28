#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/dibyanayan/tanzeel/opus testing"
OUT_DIR="/home/dibyanayan/tanzeel/14b_dataset_check/results_14b"
MODEL="qwen2.5:14b"
BASE_URL="http://localhost:11434/v1"
API_KEY="ollama"

mkdir -p "$OUT_DIR"/{strategyqa/{cot,astar},hotpotqa/{cot,astar},logicqa/{cot,astar}}

echo "Launching 14B runs: CoT + A* on StrategyQA, HotpotQA, LogicQA"

nohup python3 "$BASE_DIR/strategyqa_cot_api.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --limit 2290 \
  --output "$OUT_DIR/strategyqa/cot/results.json" \
  > "$OUT_DIR/strategyqa/cot/run.log" 2>&1 &
echo $! > "$OUT_DIR/strategyqa/cot/pid.txt"

nohup python3 "$BASE_DIR/strategyqa_astar_api.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --limit 2290 \
  --output "$OUT_DIR/strategyqa/astar/results.json" \
  > "$OUT_DIR/strategyqa/astar/run.log" 2>&1 &
echo $! > "$OUT_DIR/strategyqa/astar/pid.txt"

nohup python3 "$BASE_DIR/hotpotqa_cot_api.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --limit 2290 \
  --output "$OUT_DIR/hotpotqa/cot/results.json" \
  > "$OUT_DIR/hotpotqa/cot/run.log" 2>&1 &
echo $! > "$OUT_DIR/hotpotqa/cot/pid.txt"

nohup python3 "$BASE_DIR/hotpotqa_astar_api.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --limit 2290 \
  --output "$OUT_DIR/hotpotqa/astar/results.json" \
  > "$OUT_DIR/hotpotqa/astar/run.log" 2>&1 &
echo $! > "$OUT_DIR/hotpotqa/astar/pid.txt"

nohup python3 "$BASE_DIR/logicqa_cot_api.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --limit 651 \
  --output "$OUT_DIR/logicqa/cot/results.json" \
  > "$OUT_DIR/logicqa/cot/run.log" 2>&1 &
echo $! > "$OUT_DIR/logicqa/cot/pid.txt"

nohup python3 "$BASE_DIR/logicqa_astar_api.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --limit 651 \
  --output "$OUT_DIR/logicqa/astar/results.json" \
  > "$OUT_DIR/logicqa/astar/run.log" 2>&1 &
echo $! > "$OUT_DIR/logicqa/astar/pid.txt"

echo "All 6 jobs launched. Logs in: $OUT_DIR"
