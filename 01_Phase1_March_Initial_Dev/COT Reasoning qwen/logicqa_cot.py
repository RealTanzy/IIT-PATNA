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


def build_reasoning_fields(response: str, pred: str):
    """Build terminal-style trace and structured reasoning steps from CoT output."""
    if not response:
        terminal_trace = f"Final Answer: {pred}" if pred else ""
        return terminal_trace, []

    all_lines = [l.strip() for l in response.split('\n') if l.strip()]
    final_pat = re.compile(r'^(?:final\s+answer|the\s+answer\s+is|answer[:\s]+\[?[ABCD]\]?)\b', re.IGNORECASE)
    step_pat = re.compile(r'^(?:step\s*\d+[:.)-]?|\d+[.)-]|[-*])\s*', re.IGNORECASE)

    explicit_steps = [l for l in all_lines if step_pat.search(l) and not final_pat.search(l)]
    fallback_steps = [l for l in all_lines if not final_pat.search(l)]
    source_steps = explicit_steps if explicit_steps else fallback_steps

    reasoning_steps = []
    terminal_lines = []
    for idx, line in enumerate(source_steps, start=1):
        normalized = line if line.lower().startswith(f"step {idx}") else f"Step {idx}: {line}"
        reasoning_steps.append({
            "step_number": idx,
            "step": normalized,
            "content": line,
        })
        terminal_lines.append(normalized)

    if pred is not None:
        terminal_lines.append(f"Final Answer: {pred}")

    return "\n".join(terminal_lines), reasoning_steps

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
    ds = load_dataset("lucasmccabe/logiqa", split=args.split, streaming=True)

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
        gold_idx = int(item.get('correct_option', item.get('label', 0)))  # 0-3
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

        print(f"\nProblem {i+1}:")
        print(f"  Q: {query[:80]}...")
        print(f"  Pred: {pred} | Gold: {gold} | {'✅' if is_correct else '❌'}")

        terminal_trace, reasoning_steps = build_reasoning_fields(response, pred)

        results.append({
            "id": i,
            "context": context,
            "query": query,
            "options": options,
            "gold": gold,
            "cot_response": response,
            "trace": response,
            "terminal_trace": terminal_trace,
            "reasoning_steps": reasoning_steps,
            "pred": pred,
            "correct": is_correct,
        })

        if is_correct:
            correct_count += 1
        total += 1

    elapsed = time.time() - start_time
    acc = correct_count / total * 100 if total > 0 else 0.0

    print(f"\nFinal Accuracy: {correct_count}/{total} ({acc:.2f}%)")
    print(f"Time elapsed: {elapsed:.2f}s")

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "split": args.split,
            "accuracy": acc,
            "correct": correct_count,
            "total": total,
            "results": results,
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
