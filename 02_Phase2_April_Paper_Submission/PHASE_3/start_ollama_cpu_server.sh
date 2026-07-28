#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_NUM_GPU=0
export OLLAMA_HOST=127.0.0.1:11436

exec ollama serve
