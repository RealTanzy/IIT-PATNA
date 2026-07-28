#!/usr/bin/env bash
set -u

# Continuous Phase 2 experiment runner.
# Edits and outputs are contained in PHASE_2 as requested.

ROOT_DIR="/home/dibyanayan/tanzeel"
PHASE2_DIR="$ROOT_DIR/PHASE_2"
DATA_PATH="$ROOT_DIR/strategyqa/llama3.1_8b_42_strategyqa_astar_results.json"
OUT_ROOT="$PHASE2_DIR/results_continuous"
LOG_FILE="$PHASE2_DIR/continuous_runner.log"

CONDA_ENV="astar"
MODEL="llama3.1:8b"
BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
LIMIT_PILOT=80
LIMIT_FULL=500
SEEDS=(42 123)
HEURISTICS=(none depth coverage full)
CYCLES="${CYCLES:-1}"

mkdir -p "$OUT_ROOT"

exec >> "$LOG_FILE" 2>&1

echo "[$(date '+%F %T')] Starting Phase 2 continuous runner"
echo "Root: $ROOT_DIR"
echo "Data: $DATA_PATH"
echo "Conda env: $CONDA_ENV"
echo "Base URL: $BASE_URL"

auto_pull_model() {
  if ollama list | grep -q "^$MODEL[[:space:]]"; then
    echo "[$(date '+%F %T')] Model present: $MODEL"
  else
    echo "[$(date '+%F %T')] Model missing: $MODEL. Pulling now..."
    ollama pull "$MODEL"
  fi
}

run_one() {
  local stage="$1"
  local limit="$2"
  local seed="$3"
  local hmode="$4"

  local run_id
  run_id="${stage}_seed${seed}_${hmode}_$(date '+%Y%m%d_%H%M%S')"
  local out_json="$OUT_ROOT/${run_id}.json"

  echo "[$(date '+%F %T')] RUN $run_id"

  export CUDA_VISIBLE_DEVICES=0

  conda run -n "$CONDA_ENV" python "$PHASE2_DIR/phase2_strategyqa_astar_rewrite.py" \
    --data-path "$DATA_PATH" \
    --output "$out_json" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --limit "$limit" \
    --seed "$seed" \
    --heuristic-mode "$hmode" \
    --branch-k 3 \
    --max-depth 8 \
    --max-nodes 20 \
    --majority-vote 2 \
    --temps 0.2,0.45,0.7

  local rc=$?
  echo "[$(date '+%F %T')] DONE $run_id rc=$rc output=$out_json"
}

if [[ ! -f "$DATA_PATH" ]]; then
  echo "[$(date '+%F %T')] ERROR data path not found: $DATA_PATH"
  exit 1
fi

auto_pull_model

cycle_idx=1
while [[ "$cycle_idx" -le "$CYCLES" ]]; do
  echo "[$(date '+%F %T')] Starting cycle $cycle_idx/$CYCLES"

  # Stage 1: Pilot sweep
  for seed in "${SEEDS[@]}"; do
    for hmode in "${HEURISTICS[@]}"; do
      run_one "pilot_c${cycle_idx}" "$LIMIT_PILOT" "$seed" "$hmode"
    done
  done

  # Stage 2: Full sweep (single seed first for sustained progress)
  for hmode in "${HEURISTICS[@]}"; do
    run_one "full_c${cycle_idx}" "$LIMIT_FULL" "42" "$hmode"
  done

  cycle_idx=$((cycle_idx + 1))
done

echo "[$(date '+%F %T')] Continuous runner completed"
