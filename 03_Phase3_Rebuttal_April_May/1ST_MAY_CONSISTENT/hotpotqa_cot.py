#!/usr/bin/env python3
"""
HotpotQA CoT Solver — Fair Comparison Version
==============================================
Uses the EXACT same evaluator as hotpotqa_astar.py:
  - _normalize(): lowercase, strip articles/punct, split hyphens
  - f1_score():   token F1 with numeric equivalence
  - is_correct:   (_normalize(pred) == _normalize(gold)) OR f1 >= 0.05

Runs against VLLM (or any OpenAI-compatible endpoint).
"""

import re
import os
import json
import argparse
import time
from typing import Optional
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL     = "meta-llama/Llama-3.1-8B-Instruct"
BASE_URL  = "http://localhost:8000/v1"
API_KEY   = "none"
DEFAULT_LIMIT = 2290


# ============================================================================
# PROMPT
# ============================================================================

COT_PROMPT = """Answer the following multi-hop question by reasoning step by step.
Use the provided context paragraphs to find the answer.
At the end, output the final answer as: "The answer is: [answer]"

Context:
{context}

Question: {question}
Answer: Let's think step by step."""


# ============================================================================
# EVALUATION — identical to hotpotqa_astar.py
# ============================================================================

def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip articles/punctuation.
    MUST match hotpotqa_astar.py _normalize() exactly."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[-/]', ' ', text)                  # split hyphens/slashes
    text = re.sub(r'\b(a|an|the)\b', ' ', text)        # strip articles
    text = re.sub(r'[^\w\s]', '', text)                 # strip punctuation
    return re.sub(r'\s+', ' ', text).strip()


def _parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    s = s.strip().lower().replace(',', '')
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r'^([0-9.]+)\s*(billion|million|thousand|k|m|b)$', s)
    if m:
        n = float(m.group(1))
        suf = m.group(2)
        if suf in ('billion', 'b'):    n *= 1e9
        elif suf in ('million', 'm'):  n *= 1e6
        elif suf in ('thousand', 'k'): n *= 1e3
        return n
    return None


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 (HotpotQA official metric) — identical to A*."""
    if not pred or not gold:
        return 0.0
    p_num = _parse_number(pred)
    g_num = _parse_number(gold)
    if p_num is not None and g_num is not None:
        if abs(p_num - g_num) < 1e-3 * max(abs(p_num), abs(g_num), 1):
            return 1.0
    pred_t = _normalize(pred).split()
    gold_t = _normalize(gold).split()
    if not pred_t or not gold_t:
        return 0.0
    common = set(pred_t) & set(gold_t)
    if not common:
        return 0.0
    precision = len(common) / len(pred_t)
    recall    = len(common) / len(gold_t)
    return 2 * precision * recall / (precision + recall)


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from CoT response — grabs last 'The answer is:' match."""
    matches = re.findall(r'[Tt]he answer is[:\s]+([^\n.]+)', text)
    if matches:
        raw = matches[-1].strip().rstrip('.')
        # Trim trailing explanations (same logic as A* extract_answer)
        raw = re.split(
            r'(?:,\s*(?:but|however|although|which|because|as|and\s+(?:he|she|it|they)))',
            raw, maxsplit=1
        )[0].strip()
        raw = re.split(
            r'\s+(?:is|was|were|has|have|had)\s+(?:the|a|an)\s+',
            raw, maxsplit=1
        )[0].strip()
        return raw
    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else None


# ============================================================================
# HELPERS
# ============================================================================

def format_context(item) -> str:
    titles = item['context']['title']
    sentences_list = item['context']['sentences']
    parts = []
    for title, sentences in zip(titles, sentences_list):
        parts.append(f"{title}: {' '.join(sentences)}")
    return '\n'.join(parts)


def build_reasoning_fields(response: str, pred: Optional[str]):
    if not response:
        return (f"Final Answer: {pred}" if pred else ""), []
    all_lines  = [l.strip() for l in response.split('\n') if l.strip()]
    final_pat  = re.compile(r'^(?:final\s+answer|the\s+answer\s+is)\b', re.IGNORECASE)
    step_pat   = re.compile(r'^(?:step\s*\d+[:.)-]?|\d+[.)-]|[-*])\s*', re.IGNORECASE)
    explicit   = [l for l in all_lines if step_pat.search(l) and not final_pat.search(l)]
    source     = explicit if explicit else [l for l in all_lines if not final_pat.search(l)]
    steps, lines = [], []
    for idx, line in enumerate(source, 1):
        norm = line if line.lower().startswith(f"step {idx}") else f"Step {idx}: {line}"
        steps.append({"step_number": idx, "step": norm, "content": line})
        lines.append(norm)
    if pred:
        lines.append(f"Final Answer: {pred}")
    return "\n".join(lines), steps


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="HotpotQA CoT — Fair Comparison (same evaluator as A*)")
    parser.add_argument("--model",    default=MODEL,    help="Model name")
    parser.add_argument("--base-url", default=BASE_URL, help="API base URL")
    parser.add_argument("--api-key",  default=API_KEY,  help="API key")
    parser.add_argument("--limit",    type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--output",   default="./cot_results/hotpotqa_cot_fair.json")
    args = parser.parse_args()

    print("=" * 70)
    print("HotpotQA CoT — Fair Comparison Version")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Endpoint: {args.base_url}")
    print(f"Limit:    {args.limit}")
    print(f"Evaluator: same as A* (_normalize + f1>=0.05)")
    print("=" * 70)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results     = []
    correct     = 0
    total       = 0
    start_time  = time.time()

    def _save():
        avg_f1 = sum(r['f1'] for r in results) / total * 100 if total else 0
        acc    = correct / total * 100 if total else 0
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model,
                "accuracy": acc,
                "avg_f1": avg_f1,
                "correct": correct,
                "total": total,
                "results": results,
            }, f, indent=2)

    for i, item in tqdm(enumerate(ds), total=args.limit):
        if i >= args.limit:
            break

        q    = item['question']
        gold = item['answer']
        ctx  = format_context(item)

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": COT_PROMPT.format(question=q, context=ctx)}],
                temperature=0.2,
                max_tokens=512,
            ).choices[0].message.content or ""
        except Exception as e:
            print(f"  API error on {i}: {e}")
            response = ""

        pred = extract_answer(response)

        # ── Identical evaluator to A* ──────────────────────────────────
        f1         = f1_score(pred, gold) if pred else 0.0
        is_correct = False
        if pred:
            is_correct = (_normalize(pred) == _normalize(gold)) or f1 >= 0.05

        if is_correct:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{args.limit}] Running: {correct}/{total} "
                  f"({correct/total*100:.1f}%) | Pred: {pred} | Gold: {gold} "
                  f"| {'✅' if is_correct else '❌'} F1={f1:.2f}")

        terminal_trace, reasoning_steps = build_reasoning_fields(response, pred)
        results.append({
            "question":       q,
            "gold":           gold,
            "cot_response":   response,
            "trace":          response,
            "terminal_trace": terminal_trace,
            "reasoning_steps": reasoning_steps,
            "pred":           pred,
            "correct":        is_correct,
            "f1":             f1,
        })
        _save()

    elapsed = time.time() - start_time
    acc    = correct / total * 100 if total else 0
    avg_f1 = sum(r['f1'] for r in results) / total * 100 if total else 0
    print(f"\n{'='*70}")
    print(f"FINAL: {correct}/{total}  acc={acc:.2f}%  avg_f1={avg_f1:.2f}%")
    print(f"Time: {elapsed:.1f}s")
    print(f"Saved to {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
