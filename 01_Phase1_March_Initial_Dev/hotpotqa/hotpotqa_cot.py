#!/usr/bin/env python3
"""
Simple Chain-of-Thought (CoT) Solver for HotpotQA
"""

import os
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

COT_PROMPT = """Answer the following multi-hop question by reasoning step by step.
Use the provided context paragraphs to find the answer.
At the end, output the final answer as: "The answer is: [answer]"

Context:
{context}

Question: {question}
Answer: Let's think step by step."""

# ============================================================================
# MAIN
# ============================================================================

def extract_answer(text: str) -> str:
    # Grab ALL matches and return the last one (model often repeats)
    matches = re.findall(r'[Tt]he answer is[: ]+([^.\n]+)', text)
    if matches:
        return matches[-1].strip().rstrip('.')
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else None


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 — HotpotQA official metric."""
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


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles for EM comparison."""
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def format_context(item) -> str:
    """Format HotpotQA context paragraphs into a readable string."""
    titles = item['context']['title']
    sentences_list = item['context']['sentences']
    parts = []
    for title, sentences in zip(titles, sentences_list):
        para = ' '.join(sentences)
        parts.append(f"{title}: {para}")
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model")
    parser.add_argument("--limit", type=int, default=500, help="Number of examples")
    parser.add_argument("--output", default="./results_shot/llama3.1_8b_hotpotqa_cot_results.json", help="Output file")
    args = parser.parse_args()

    print(f"Running CoT on HotpotQA with model: {args.model}")

    solver = OllamaClient(args.model)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)

    results = []
    correct_count = 0
    total = 0

    start_time = time.time()

    for i, item in tqdm(enumerate(ds)):
        if i >= args.limit: break

        q = item['question']
        gold = item['answer']
        ctx = format_context(item)

        # Run CoT
        prompt = COT_PROMPT.format(question=q, context=ctx)
        response = solver.generate(prompt)

        # Extract
        pred = extract_answer(response)

        # Evaluate (normalized exact match + F1)
        f1 = f1_score(pred, gold) if pred else 0.0
        is_correct = (normalize(pred) == normalize(gold)) or f1 > 0.3

        print(f"Problem {i+1}:")
        print(f"Q: {q[:60]}...")
        print(f"Pred: {pred} | Gold: {gold} | EM: {'✅' if is_correct else '❌'} | F1: {f1:.2f}")

        results.append({
            "question": q,
            "gold": gold,
            "cot_response": response,
            "pred": pred,
            "correct": is_correct,
            "f1": f1
        })

        if is_correct: correct_count += 1
        total += 1

    elapsed = time.time() - start_time
    acc = correct_count / total * 100
    avg_f1 = sum(r['f1'] for r in results) / total * 100
    print(f"Results: {correct_count}/{total} EM={acc:.2f}%  F1={avg_f1:.2f}%")
    print(f"Time: {elapsed:.2f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
