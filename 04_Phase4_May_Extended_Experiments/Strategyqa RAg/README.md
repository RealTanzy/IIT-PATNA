# Strategyqa RAg

A clean StrategyQA retrieval-augmented reasoning pipeline based on CoT + A* correct traces.

## What This Implements

1. Build a paired StrategyQA file from CoT and A* run outputs.
2. Build a reasoning knowledge base from paired runs.
3. Keep CoT reasoning where CoT is correct.
4. Keep A* reasoning where A* is correct.
5. Store entries with:
   - id
   - question
   - reasoning
   - answer (yes/no)
6. Build a FAISS index for cosine-style retrieval over question text.
7. For a new query, retrieve the highest-similarity question(s) and use retrieved reasoning as generation context.

## Files

- build_strategyqa_paired_jsonl.py
  - Builds paired StrategyQA JSONL from CoT + A* JSON results.
- build_strategyqa_faiss_kb.py
  - Builds KB JSONL files and FAISS index.
- strategyqa_faiss_rag.py
  - Runs query-time retrieval and yes/no answer generation.
- evaluate_strategyqa_faiss_rag_batch.py
  - Runs batch evaluation and writes predictions + metrics.
- artifacts/
  - Generated outputs.

## 1) Build Paired StrategyQA JSONL

python3 "Strategyqa RAg/build_strategyqa_paired_jsonl.py" \
  --astar-file "FINAL_ASTAR/qwen_results/qwen0.5b_strategyqa_astar_results.json" \
  --cot-file "COT Reasoning qwen/qwen_results/qwen0.5b_strategyqa_cot_results.json" \
  --out-jsonl "Strategyqa RAg/data/cot_astar_runs/strategyqa_paired_all.jsonl"

Generated files:
- Strategyqa RAg/data/cot_astar_runs/strategyqa_paired_all.jsonl
- Strategyqa RAg/data/cot_astar_runs/summary.json

## 2) Build KB + FAISS Index

python3 "Strategyqa RAg/build_strategyqa_faiss_kb.py" \
  --paired-jsonl "Strategyqa RAg/data/cot_astar_runs/strategyqa_paired_all.jsonl" \
  --output-dir "Strategyqa RAg/artifacts" \
  --split all

Generated files:
- Strategyqa RAg/artifacts/strategyqa_reasoning_kb.jsonl
- Strategyqa RAg/artifacts/strategyqa_reasoning_kb_cot.jsonl
- Strategyqa RAg/artifacts/strategyqa_reasoning_kb_astar.jsonl
- Strategyqa RAg/artifacts/strategyqa_questions.faiss
- Strategyqa RAg/artifacts/strategyqa_index_meta.json

## 3) Query RAG (one-shot)

python3 "Strategyqa RAg/strategyqa_faiss_rag.py" \
  --query "Did the Roman Empire exist before smartphones?" \
  --top-k 1

## 4) Query RAG (interactive)

python3 "Strategyqa RAg/strategyqa_faiss_rag.py" --interactive

## 5) Retrieval-only check

python3 "Strategyqa RAg/strategyqa_faiss_rag.py" \
  --query "Did the Roman Empire exist before smartphones?" \
  --top-k 1 \
  --retrieval-only

## 6) Batch evaluation

python3 "Strategyqa RAg/evaluate_strategyqa_faiss_rag_batch.py" \
  --dataset-jsonl "Strategyqa RAg/data/cot_astar_runs/strategyqa_paired_all.jsonl" \
  --split all \
  --max-questions 100 \
  --top-k 1 \
  --output-dir "strategyqa test output"
