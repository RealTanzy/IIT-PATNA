#!/usr/bin/env bash
set -u

# Wrapper to run continuous experiments against GPU0-bound Ollama server.

export OLLAMA_BASE_URL="http://127.0.0.1:11435/v1"
export CYCLES="${CYCLES:-2}"

bash /home/dibyanayan/tanzeel/PHASE_2/run_phase2_continuous.sh
