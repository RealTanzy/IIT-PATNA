# Logicsqa RAg

A clean LogicQA retrieval-augmented reasoning pipeline based on CoT + A* correct traces.

## What This Implements

1. Build a reasoning knowledge base from prior paired runs.
2. Keep CoT reasoning where CoT is correct.
3. Keep A* reasoning where A* is correct.
4. Store entries with:
   - id
   - question
   - reasoning
   - answer
5. Build a FAISS index for cosine-style retrieval over question text.
6. For a new query, retrieve highest-similarity question(s) and use retrieved reasoning as generation context.

## Files

- build_logicqa_faiss_kb.py
  - Builds KB JSONL files and FAISS index.
- logicqa_faiss_rag.py
  - Runs query-time retrieval and answer generation.
- artifacts/
  - Generated outputs.

## Build KB + FAISS Index

python3 "Logicsqa RAg/build_logicqa_faiss_kb.py" \
  --paired-jsonl "THE FINAL CALL/llama 3.1 8b/data/cot_astar_runs/logicqa_paired_all.jsonl" \
  --output-dir "Logicsqa RAg/artifacts" \
  --split all

Generated files:
- Logicsqa RAg/artifacts/logicqa_reasoning_kb.jsonl
- Logicsqa RAg/artifacts/logicqa_reasoning_kb_cot.jsonl
- Logicsqa RAg/artifacts/logicqa_reasoning_kb_astar.jsonl
- Logicsqa RAg/artifacts/logicqa_questions.faiss
- Logicsqa RAg/artifacts/logicqa_index_meta.json

## Query RAG (one-shot)

python3 "Logicsqa RAg/logicqa_faiss_rag.py" \
  --query "Which option best weakens the argument?" \
  --context "A study claims X because Y." \
  --options "Option one|||Option two|||Option three|||Option four" \
  --top-k 1

## Query RAG (interactive)

python3 "Logicsqa RAg/logicqa_faiss_rag.py" --interactive

## Retrieval-only check

python3 "Logicsqa RAg/logicqa_faiss_rag.py" \
  --query "Which option best weakens the argument?" \
  --context "A study claims X because Y." \
  --options "Option one|||Option two|||Option three|||Option four" \
  --top-k 1 \
  --retrieval-only
