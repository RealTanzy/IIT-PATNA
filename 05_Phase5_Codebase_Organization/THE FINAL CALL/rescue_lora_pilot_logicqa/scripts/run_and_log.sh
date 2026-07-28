#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 day1|day2 <command> [args ...]"
  exit 2
fi

DAY="$1"
shift

if [[ "$DAY" != "day1" && "$DAY" != "day2" ]]; then
  echo "First argument must be day1 or day2"
  exit 2
fi

LOG_MD="$ROOT/notes/${DAY}_log.md"
RUN_TS="$(date '+%Y-%m-%d %H:%M:%S %z')"
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG="$ROOT/outputs/metrics/live_${DAY}_${RUN_TAG}.log"

mkdir -p "$ROOT/outputs/metrics"

echo "" >> "$LOG_MD"
echo "### Live Entry: $RUN_TS" >> "$LOG_MD"
echo "- Command: $*" >> "$LOG_MD"
echo "- Run log: outputs/metrics/$(basename "$RUN_LOG")" >> "$LOG_MD"

set +e
"$@" > >(tee "$RUN_LOG") 2>&1
EC=$?
set -e

END_TS="$(date '+%Y-%m-%d %H:%M:%S %z')"
if [[ $EC -eq 0 ]]; then
  STATUS="success"
else
  STATUS="failed"
fi

echo "- End time: $END_TS" >> "$LOG_MD"
echo "- Exit code: $EC" >> "$LOG_MD"
echo "- Status: $STATUS" >> "$LOG_MD"

exit $EC
