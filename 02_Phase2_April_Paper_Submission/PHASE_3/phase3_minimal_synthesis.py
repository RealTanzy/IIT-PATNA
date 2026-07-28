#!/usr/bin/env python3
"""Phase 3 Final: Minimal synthesis prompts to avoid Ollama crashes."""

import argparse
import json
import re
import time
from typing import List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from phase3_prompts import (
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
)


class Phase3FinalRunner:
    def __init__(self, model: str, base_url: str, max_trace_chars: int = 400):
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=120, max_retries=0)
        self.model = model
        self.max_trace_chars = max_trace_chars
        self.api_calls = 0
        self.api_successes = 0
        self.api_failures = 0

    def _truncate(self, text: str) -> str:
        if not text or len(text) <= self.max_trace_chars:
            return text
        return text[:self.max_trace_chars] + "..."

    def _extract_answer(self, text: str) -> Optional[str]:
        if not text:
            return None
        # Look for Final Answer
        m = re.search(r"(?:Final\s*)?[Aa]nswer\s*:\s*([A-D])", text)
        if m:
            return m.group(1).upper()
        # Look for option letter
        m = re.search(r"\b([A-D])\b", text[-200:])
        if m:
            return m.group(1).upper()
        return None

    def synthesize(self, question: str, cot: str, astar: str) -> Tuple[str, Optional[str], str]:
        """Minimal synthesis attempt."""
        try:
            self.api_calls += 1
            # Extremely short prompt
            short_cot = self._truncate(cot)
            short_astar = self._truncate(astar)
            
            prompt = f"Question: {question[:200]}\n\nCoT: {short_cot}\n\nA*: {short_astar}\n\nAnswer: "
            
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=50,
            )
            result = resp.choices[0].message.content.strip()
            self.api_successes += 1
            return result, self._extract_answer(result), "llm"
            
        except Exception as e:
            self.api_failures += 1
            #Fallback
            combined = f"{cot}\n{astar}"
            return combined, self._extract_answer(combined), "fallback"


def run(args: argparse.Namespace) -> None:
    with open(args.astar_path) as f:
        astar_data = json.load(f)
    with open(args.cot_path) as f:
        cot_data = json.load(f)

    astar_results = astar_data.get("results", [])
    cot_by_id = {int(r["id"]): r for r in cot_data.get("results", []) if "id" in r}

    runner = Phase3FinalRunner(args.model, args.base_url, args.max_trace_chars)

    merged_rows = []
    correct = 0

    for arow in tqdm(astar_results, desc="Phase 3 minimal"):
        qid = int(arow.get("id", -1))
        crow = cot_by_id.get(qid, {})

        gold = str(arow.get("gold") or crow.get("gold") or "").strip()
        query = arow.get("query") or crow.get("query") or ""
        cot_trace = crow.get("reasoning_trace") or crow.get("trace") or ""
        astar_trace = arow.get("reasoning_trace") or arow.get("trace") or ""

        synth_text, pred, source = runner.synthesize(query, cot_trace, astar_trace)
        is_correct = (pred == gold) if pred and gold else False
        if is_correct:
            correct += 1

        merged_rows.append({
            "id": qid,
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "query": query,
            "cot_trace": cot_trace,
            "astar_trace": astar_trace,
            "synthesis_source": source,
        })

    output = {
        "phase": "phase3_minimal_synthesis",
        "model": runner.model,
        "total": len(merged_rows),
        "correct": correct,
        "accuracy": (correct / len(merged_rows) * 100) if merged_rows else 0,
        "api_calls": runner.api_calls,
        "api_successes": runner.api_successes,
        "api_failures": runner.api_failures,
        "results": merged_rows,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"✓ Phase 3 complete")
    print(f"  Model: {runner.model}")
    print(f"  Total: {output['total']}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {output['accuracy']:.1f}%")
    print(f"  LLM successes: {runner.api_successes}")
    print(f"  LLM failures: {runner.api_failures}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--astar-path", required=True)
    parser.add_argument("--cot-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--max-trace-chars", type=int, default=400)
    args = parser.parse_args()
    run(args)
