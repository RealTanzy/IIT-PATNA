#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
exec python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 4096 \
    --dtype bfloat16
