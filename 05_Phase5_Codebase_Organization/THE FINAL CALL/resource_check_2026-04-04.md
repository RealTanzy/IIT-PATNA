# Resource Check (April 4, 2026)

## Confirmed Compute Resources
- Conda environment available: astar
- GPU available: NVIDIA A100 80GB PCIe
- Verified device 0:
  - index: 0
  - memory total: 81920 MiB
  - memory used at check: 3533 MiB
  - memory free at check: 77621 MiB
  - compute mode: Default
- Additional GPU present: device 1 (NVIDIA A100 80GB PCIe)

## LLM Models Detected (Ollama)
- llama3.1:8b
- qwen3-coder-next:latest

## Python Runtime Check inside astar env
- python runs in astar env: yes
- torch installed in astar env: no
- implication: CUDA is available at system level (nvidia-smi), but torch-based CUDA probing is not usable unless torch is installed in astar env.

## Working Assumption for Next Steps
- We can run experiments on GPU device 0.
- We can use available local LLM models via Ollama.
- We should not rely on torch inside astar env unless we install it later.
