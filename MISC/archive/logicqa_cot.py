#!/usr/bin/env python3
"""
Simple Chain-of-Thought (CoT) Solver for LogicQA
Adapted from GSM8K CoT solver for multiple-choice logical reasoning.
"""

import re
import json
import argparse
import requests
import time
from datasets import load_dataset
from tqdm import tqdm

# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    def __init__(self, model: str, base_url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "stop": ["Question:", "Context:"]
            }
        }
        try:
            res = requests.post(self.base_url, json=data)
            if res.status_code == 200:
                return res.json().get('response', '')
            else:
                return ""
        except Exception as e:
            print(f"Error: {e}")
            return ""

# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

COT_PROMPT = """Read the passage carefully and answer the multiple-choice question by reasoning step by step.
At the end, output the final answer as: "The answer is [A/B/C/D]"

Passage: {context}

Question: {query}

Options:
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Answer: Let's think step by step."""

# ============================================================================
# HELPERS
# ============================================================================

LABEL_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

def extract_answer(text: str) -> str:
    """Extract the predicted option letter (A/B/C/D) from model output."""
    # Primary: "The answer is [X]" or "The answer is X"
    match = re.search(r'[Tt]he answer is\s*\[?([ABCD])\]?', text)
    if match:
        return match.group(1).upper()

    # Secondary: "answer: X" or "answer is X"
    match = re.search(r'answer[:\s]+\[?([ABCD])\]?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Tertiary: standalone letter at end of text
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        match = re.fullmatch(r'\[?([ABCD])\]?\.?', line)
        if match:
            return match.group(1).upper()

    # Last resort: first occurrence of a lone A/B/C/D
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1).upper()

    return None

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# ============================================================================
# MAIN
# ============================================================================

def parse_limit(value: str):
    """Accept an integer or 'all' for --limit."""
    if value.lower() == 'all':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--limit must be an integer or 'all', got: {value!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b", help="Ollama model name")
    parser.add_argument("--limit", type=parse_limit, default=10,
                        help="Number of examples to evaluate (integer) or 'all' for the full split")
    parser.add_argument("--split", default="test", help="Dataset split: train / test / validation")
    parser.add_argument("--output", default="./results_shot/qwen_logicqa_cot_results.json",
                        help="Path to output JSON file")
    args = parser.parse_args()

    print(f"Running CoT on LogicQA with model: {args.model}")

    solver = OllamaClient(args.model)

    # lucasmccabe/logiqa  fields: label (int 0-3), context, query, options (list of 4)
    ds = load_dataset("lucasmccabe/logiqa", split=args.split, streaming=True, trust_remote_code=True)

    results = []
    correct_count = 0
    total = 0
    start_time = time.time()

    for i, item in tqdm(enumerate(ds)):
        if args.limit is not None and i >= args.limit:
            break

        context  = item['context']
        query    = item['query']
        options  = item['options']          # list of 4 strings
        gold_idx = int(item['correct_option'])  # 0-3
        gold     = LABEL_MAP[gold_idx]      # "A"/"B"/"C"/"D"

        prompt = COT_PROMPT.format(
            context=context,
            query=query,
            opt_a=options[0],
            opt_b=options[1],
            opt_c=options[2],
            opt_d=options[3],
        )

        response = solver.generate(prompt)
        pred     = extract_answer(response)
        is_correct = (pred == gold)

        pred_idx  = ord(pred) - ord('A') if pred else None
        pred_text = options[pred_idx] if pred_idx is not None and 0 <= pred_idx < len(options) else (pred or "")
        gold_text = options[gold_idx]
        f1 = f1_score(pred_text, gold_text)

        print(f"\nProblem {i+1}:")
        print(f"  Q: {query[:80]}...")
        print(f"  Pred: {pred} | Gold: {gold} | {'✅' if is_correct else '❌'} | F1: {f1:.2f}")

        results.append({
            "id": i,
            "context": context,
            "query": query,
            "options": options,
            "gold": gold,
            "cot_response": response,
            "pred": pred,
            "correct": is_correct,
            "f1": f1,
        })

        if is_correct:
            correct_count += 1
        total += 1

    elapsed = time.time() - start_time
    acc     = correct_count / total * 100 if total > 0 else 0.0
    avg_f1  = sum(r['f1'] for r in results) / total * 100 if total > 0 else 0.0

    print(f"\nFinal Accuracy: {correct_count}/{total} ({acc:.2f}%)  |  Avg F1: {avg_f1:.1f}%")
    print(f"Time elapsed: {elapsed:.2f}s")

    import os
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Merge with existing file if present
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing = json.load(f)
        id_offset = len(existing.get("results", []))
        for r in results:
            r["id"] += id_offset
        merged_results  = existing.get("results", []) + results
        merged_correct  = existing.get("correct", 0) + correct_count
        merged_total    = existing.get("total",   0) + total
        merged_acc      = merged_correct / merged_total * 100 if merged_total > 0 else 0.0
        merged_avg_f1   = sum(r["f1"] for r in merged_results) / merged_total * 100 if merged_total > 0 else 0.0
        print(f"Appending to existing file ({id_offset} prior results).")
    else:
        merged_results = results
        merged_correct = correct_count
        merged_total   = total
        merged_acc     = acc
        merged_avg_f1  = avg_f1

    with open(args.output, "w") as f:
        json.dump({
            "model":   args.model,
            "split":   args.split,
            "accuracy": merged_acc,
            "avg_f1":  merged_avg_f1,
            "correct": merged_correct,
            "total":   merged_total,
            "results": merged_results,
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
