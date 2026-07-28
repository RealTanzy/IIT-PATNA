#!/usr/bin/env bash
set -u

# Starts a dedicated Ollama server pinned to GPU 0 on port 11435.
# This avoids interfering with any existing default Ollama service.

LOG_FILE="/home/dibyanayan/tanzeel/PHASE_2/ollama_gpu0_server.log"

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_HOST=127.0.0.1:11435

echo "[$(date '+%F %T')] Starting Ollama GPU0 server at $OLLAMA_HOST" >> "$LOG_FILE"
ollama serve >> "$LOG_FILE" 2>&1
