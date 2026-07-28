#!/usr/bin/env python3
"""
Simple Chain-of-Thought (CoT) Solver for StrategyQA
Yes/No questions answered using world knowledge with step-by-step reasoning.
"""

import re
import json
import os
import argparse
import requests
import time
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
                "stop": ["Question:"]
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

COT_PROMPT = """Answer the following yes/no question by reasoning step by step from your world knowledge.
At the end, output the final answer as: "Final Answer: Yes" or "Final Answer: No"

Question: {question}
Answer: Let's think step by step."""

# ============================================================================
# HELPERS
# ============================================================================

def extract_answer(text: str) -> str:
    """Extract yes/no from a CoT response."""
    # 'Final Answer: Yes/No' pattern (with optional markdown bold)
    m = re.search(r'[Ff]inal\s+[Aa]nswer[:\s*]+\**(yes|no)\**', text, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 'The answer is: yes/no'
    m = re.search(r'[Tt]he answer is[:\s]+(yes|no)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Bare yes/no on last line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        last = lines[-1].lower()
        if re.search(r'\byes\b', last):
            return 'yes'
        if re.search(r'\bno\b', last):
            return 'no'

    return None


def build_reasoning_fields(response: str, pred: str):
    """Build terminal-style trace and structured reasoning steps from CoT output."""
    if not response:
        terminal_trace = f"Final Answer: {pred}" if pred else ""
        return terminal_trace, []

    all_lines = [l.strip() for l in response.split('\n') if l.strip()]
    final_pat = re.compile(r'^(?:final\s+answer|the\s+answer\s+is)\b', re.IGNORECASE)
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


def load_strategyqa(local_path=None):
    """
    Load StrategyQA dataset.
    Priority:
      1. local_path if provided
      2. HuggingFace hub (wics/strategy-qa)

    Returns list of dicts with keys: 'question', 'answer' (str: 'yes'/'no').
    """
    if local_path and os.path.exists(local_path):
        print(f"Loading StrategyQA from local file: {local_path}")
        with open(local_path) as f:
            raw = json.load(f)
        items = []
        for entry in raw:
            gold = entry.get("gold", entry.get("answer", entry.get("label")))
            if gold is None:
                continue
            if isinstance(gold, bool):
                gold_str = "yes" if gold else "no"
            else:
                gold_str = str(gold).lower().strip()
            items.append({
                "question": entry["question"],
                "answer": gold_str,
            })
        print(f"Loaded {len(items)} examples")
        return items

    # Fallback: HuggingFace
    print("Loading StrategyQA from HuggingFace (wics/strategy-qa)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wics/strategy-qa", split="test")
        items = []
        for entry in ds:
            gold = entry.get("answer", entry.get("label"))
            if isinstance(gold, bool):
                gold_str = "yes" if gold else "no"
            else:
                gold_str = str(gold).lower().strip()
            items.append({
                "question": entry["question"],
                "answer": gold_str,
            })
        print(f"Loaded {len(items)} examples from HuggingFace")
        return items
    except Exception as e:
        print(f"HuggingFace load failed: {e}")
        raise RuntimeError(
            "Could not load StrategyQA dataset. "
            "Provide --data-path or ensure wics/strategy-qa is accessible."
        )


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
    parser = argparse.ArgumentParser(description="StrategyQA CoT Solver (yes/no, world knowledge)")
    parser.add_argument("--model", default="qwen2.5:0.5b", help="Ollama model name")
    parser.add_argument("--limit", type=parse_limit, default=500,
                        help="Number of examples to evaluate (integer) or 'all' for the full split")
    parser.add_argument("--data-path", default=None,
                        help="Path to local StrategyQA JSON file")
    parser.add_argument("--output", default="./results/strategyqa_cot_results.json",
                        help="Path to output JSON file")
    args = parser.parse_args()

    print(f"Running CoT on StrategyQA with model: {args.model}")

    solver = OllamaClient(args.model)
    dataset = load_strategyqa(args.data_path)

    if args.limit is not None:
        dataset = dataset[:args.limit]

    results = []
    correct_count = 0
    total = 0
    start_time = time.time()

    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        question = item['question']
        gold = item['answer']

        prompt = COT_PROMPT.format(question=question)
        response = solver.generate(prompt)
        pred = extract_answer(response)
        is_correct = (pred == gold) if pred else False

        print(f"\nProblem {i+1}:")
        print(f"  Q: {question[:80]}...")
        print(f"  Pred: {pred} | Gold: {gold} | {'✅' if is_correct else '❌'}")

        terminal_trace, reasoning_steps = build_reasoning_fields(response, pred)

        results.append({
            "id": i,
            "question": question,
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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "accuracy": acc,
            "correct": correct_count,
            "total": total,
            "results": results,
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
