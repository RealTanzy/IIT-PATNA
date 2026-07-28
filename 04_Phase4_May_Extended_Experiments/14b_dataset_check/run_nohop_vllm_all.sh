#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/dibyanayan/tanzeel/19th_April/TMLR_Version/Rebuttal_Experiments"
OUT_DIR="/home/dibyanayan/tanzeel/14b_dataset_check/results_nohop_14b"

# Ollama OpenAI-compatible endpoint + selected 14B model
BASE_URL="http://localhost:11434/v1"
MODEL_NAME="qwen2.5:14b"

mkdir -p "$OUT_DIR"

echo "[1/3] HotpotQA no-hop (K=1) on vLLM"
python3 "$ROOT_DIR/hotpotqa_ablation.py" \
	--sample-file "$ROOT_DIR/samples/hotpotqa_300.json" \
	--heuristic-mode astar \
	--branch-k 1 \
	--budget 15 \
	--model "$MODEL_NAME" \
	--base-url "$BASE_URL" \
	--out "$OUT_DIR/hotpotqa_nohop_vllm.json"

echo "[2/3] LogicQA no-hop (K=1) on vLLM"
python3 "$ROOT_DIR/logicqa_ablation.py" \
	--sample-file "$ROOT_DIR/samples/logicqa_200.json" \
	--heuristic-mode astar \
	--branch-k 1 \
	--budget 12 \
	--model "$MODEL_NAME" \
	--base-url "$BASE_URL" \
	--out "$OUT_DIR/logicqa_nohop_vllm.json"

echo "[3/3] StrategyQA no-hop (K=1) on vLLM"
python3 "$ROOT_DIR/strategyqa_ablation.py" \
	--sample-file "$ROOT_DIR/samples/strategyqa_200.json" \
	--heuristic-mode astar \
	--branch-k 1 \
	--budget 15 \
	--model "$MODEL_NAME" \
	--base-url "$BASE_URL" \
	--out "$OUT_DIR/strategyqa_nohop_vllm.json"

echo "Done. Results written to: $OUT_DIR"
