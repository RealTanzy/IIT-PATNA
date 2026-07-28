#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa"

echo "[1/4] Build rescue/preserve buckets"
python "$ROOT/scripts/build_rescue_set.py"

echo "[2/4] Rewrite rescue rationales with local LLM"
python "$ROOT/scripts/rewrite_rationales.py"

echo "[3/4] Filter weak rewrites"
python "$ROOT/scripts/filter_rationales.py"

echo "[4/4] Build LoRA train/dev JSONL"
python "$ROOT/scripts/make_lora_jsonl.py"

echo "Done. Data artifacts are ready under: $ROOT/data"
