# A* to CoT Rescue LoRA Pilot (LogicQA, 48h Minimal)

This pilot implements the exact rescue-distillation idea from your PDF plan:
- Focus on CoT failures where A* succeeds.
- Rewrite successful A* traces into short linear rationales.
- Train a compact LoRA adapter on mixed data:
  - preserve (CoT-correct)
  - rescue (A*-only corrected)
  - small answer-only slice
- Evaluate with four checks:
  - overall accuracy
  - rescue rate
  - preservation rate
  - category-wise repair analysis

## Important Data Note
The StrategyQA 100-sample files in this workspace are mostly not paired (very low overlap between CoT and A* questions), so this pilot uses LogicQA 100-sample paired files where overlap is clean.

## Source Files Used
- A*: `Tanzeel_astar/Dibyanayan_results/logicqa_astar_100_llama3.1_8b_device0.json`
- CoT: `Tanzeel_astar/Dibyanayan_results/logicqa_cot_100_llama3.1_8b_device0.json`

## Directory Map
- `data/`: built datasets and manifests
- `prompts/`: rewrite and verification prompts
- `scripts/`: full pipeline scripts
- `outputs/`: adapters, generations, metrics
- `notes/`: day logs

## Quick Start
1. Build paired buckets and manifests:
   - `python scripts/build_rescue_set.py`
2. Rewrite rescue traces with local LLM:
   - `python scripts/rewrite_rationales.py`
3. Filter weak rewrites:
   - `python scripts/filter_rationales.py`
4. Build LoRA train/dev JSONL:
   - `python scripts/make_lora_jsonl.py`
5. Train adapter (requires `peft`):
   - `CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py --base-model Qwen/Qwen2.5-0.5B-Instruct`
6. Evaluate 4 checks:
   - `CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_rescue.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path outputs/adapters/rescue_lora_logicqa/final`

## One-command Data Pipeline
- `bash scripts/run_minimal_pipeline.sh`

This runs steps 1-4 (data construction + rewrite + filtering + training file creation).
