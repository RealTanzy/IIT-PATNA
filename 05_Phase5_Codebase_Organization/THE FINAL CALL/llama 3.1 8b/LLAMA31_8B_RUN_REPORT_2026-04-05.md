# Llama 3.1 8B Rescue LoRA Run Report

Date: 2026-04-05
Folder: THE FINAL CALL/llama 3.1 8b

## 1) What Was Requested
Run the same rescue-distillation pipeline for Llama 3.1 8B and keep outputs in a separate folder.

## 2) Pipeline Executed
Using scripts from:
- THE FINAL CALL/rescue_lora_pilot_logicqa/scripts/

Executed in new folder:
- build_rescue_set.py
- rewrite_rationales.py
- filter_rationales.py
- make_lora_jsonl.py
- train_lora.py
- evaluate_rescue.py

## 3) Data Summary
- total overlap: 100
- bucket counts: both_correct=20, cot_only=19, astar_only=13, both_wrong=48
- selected rescue train: 10
- selected preserve train: 31
- selected answer-only train: 6
- train records: 47
- dev records: 11
- rewritten rescues kept: 13/13

## 4) Training Configuration
- base model: meta-llama/Llama-3.1-8B-Instruct
- output adapter dir: outputs/adapters/rescue_lora_llama31_8b_fix1
- epochs: 1.3
- learning rate: 5e-5
- batch size: 2
- grad accumulation: 4
- max length: 512
- LoRA defaults: r=16, alpha=16, dropout=0.05

## 5) Evaluation Results (dev n=21)
From outputs/metrics/rescue_eval_llama31_8b_fix1.json:
- overall_accuracy: 33.33
- rescue_rate: 33.33
- preservation_rate: 50.00
- rescue_cases: 3
- preserve_cases: 8
- category-wise repair (logical_constraints): 1/3 (33.33)

## 6) Comparison to Current Best Qwen Run
Qwen best (Fix-3) from rescue_lora_pilot_logicqa:
- overall_accuracy: 28.57
- rescue_rate: 33.33
- preservation_rate: 25.00

Llama 3.1 8B run in this folder:
- overall_accuracy: 33.33
- rescue_rate: 33.33
- preservation_rate: 50.00

Delta (Llama - Qwen):
- overall: +4.76
- rescue: +0.00
- preservation: +25.00

## 7) Fine-Tuned Model Location
Best adapter produced in this run:
- absolute path:
  /home/dibyanayan/tanzeel/THE FINAL CALL/llama 3.1 8b/outputs/adapters/rescue_lora_llama31_8b_fix1/final
- main weights file:
  /home/dibyanayan/tanzeel/THE FINAL CALL/llama 3.1 8b/outputs/adapters/rescue_lora_llama31_8b_fix1/final/adapter_model.safetensors

## 8) Key Artifacts
- outputs/metrics/build_llama31.log
- outputs/metrics/rewrite_llama31.log
- outputs/metrics/filter_llama31.log
- outputs/metrics/make_llama31.log
- outputs/metrics/train_llama31.log
- outputs/metrics/eval_llama31.log
- outputs/metrics/rescue_eval_llama31_8b_fix1.json
- outputs/metrics/sample_outputs_llama31_8b_fix1.csv
