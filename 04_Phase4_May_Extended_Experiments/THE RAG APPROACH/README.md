# LogicQA Complementary RAG (CoT + A*)

This folder now contains a full dataset-wise RAG experiment for LogicQA using Llama 3.1 8B.

## Core Idea
Use complementary reasoning strengths from existing runs:
- `cot_only` (CoT correct, A* wrong): preserve CoT reasoning
- `astar_only` (A* correct, CoT wrong): linearize A* reasoning into compact step-wise form

These become a reasoning knowledge base (KB) for retrieval-augmented answering.

## Implemented Script
- `logicqa_rag_experiment.py`
- `logicqa_rag_query_test.py` (single-query / interactive tester)

## What The Script Does
1. Load paired LogicQA data (`logicqa_paired_all.jsonl`)
2. Build complementary KB entries with fields:
   - `id`
   - `question`
   - `context` (linearized reasoning path + final answer)
   - `style` and source metadata
3. Build cosine retriever over stored questions:
   - `sentence-transformers` embeddings (default)
   - falls back to TF-IDF if needed
4. For each eval query:
   - retrieve top-k similar KB examples
   - prompt Llama 3.1 8B with retrieved contexts
   - generate answer
5. Save predictions and accuracy metrics

## Main Commands

### 1) Build KB only (smoke check)
```bash
python3 "THE RAG APPROACH/logicqa_rag_experiment.py" \
  --build-kb-only \
  --kb-split train \
  --output-dir "THE RAG APPROACH/outputs/logicqa_rag_smoke"
```

### 2) Full dataset-wise evaluation (run used)
```bash
CUDA_VISIBLE_DEVICES=0 python3 "THE RAG APPROACH/logicqa_rag_experiment.py" \
  --kb-split train \
  --eval-split all \
  --top-k 3 \
  --max-new-tokens 140 \
  --output-dir "THE RAG APPROACH/outputs/logicqa_rag_run2"
```

### 3) Strict holdout-style dev evaluation (run used)
```bash
CUDA_VISIBLE_DEVICES=0 python3 "THE RAG APPROACH/logicqa_rag_experiment.py" \
  --kb-split train \
  --eval-split dev \
  --top-k 3 \
  --max-new-tokens 140 \
  --output-dir "THE RAG APPROACH/outputs/logicqa_rag_run2_dev"
```

### 4) Test one custom query (single .py)
```bash
CUDA_VISIBLE_DEVICES=0 python3 "THE RAG APPROACH/logicqa_rag_query_test.py" \
  --query "Which of the following best refutes the argument?" \
  --context "The argument assumes sales directly equals popularity." \
  --options "Sales only partially reflects popularity|||Buyers are educated|||There are many types of life books|||Some sold books are unread"
```

### 5) Interactive query testing
```bash
CUDA_VISIBLE_DEVICES=0 python3 "THE RAG APPROACH/logicqa_rag_query_test.py" --interactive
```

## Tests

Test file:
- `THE RAG APPROACH/tests/test_logicqa_rag_experiment.py`

Run tests:
```bash
pytest -q "THE RAG APPROACH/tests/test_logicqa_rag_experiment.py"
```

Current test coverage includes:
- final-answer extraction variants
- A* trace linearization and dedup behavior
- complementary KB construction on train split
- retriever top-k behavior with self-exclusion
- CLI build-kb-only smoke run artifact checks

## Key Output Files Per Run
- `complementary_kb.jsonl`
- `complementary_kb.json`
- `kb_summary.json`
- `eval_predictions.jsonl`
- `eval_metrics.json`
- `run_config.json`

## Default Inputs
- Paired LogicQA file:
  - `/home/dibyanayan/tanzeel/THE FINAL CALL/llama 3.1 8b/data/cot_astar_runs/logicqa_paired_all.jsonl`
- Base model:
  - `meta-llama/Llama-3.1-8B-Instruct`
