#!/usr/bin/env python3
"""Phase 3 v2: Robust rewrite + synthesis with better error handling."""

import argparse
import json
import os
import re
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from phase3_prompts import (
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
)


@dataclass
class Phase3Stats:
    total: int = 0
    correct: int = 0
    api_calls: int = 0
    total_tokens: int = 0
    rewrite_failures: int = 0
    synthesis_failures: int = 0
    synthesis_retries: int = 0


class Phase3DualTraceRunnerV2:
    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        max_trace_chars: int = 1000,  # AGGRESSIVE DEFAULT (was 3500)
        rewrite_mode: str = "deterministic",
        use_fallbacks: bool = False,
        request_timeout: float = 60.0,
        verbose: bool = False,
        max_retries: int = 3,
        enable_warmup: bool = True,
    ):
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=request_timeout, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_trace_chars = max_trace_chars
        self.rewrite_mode = rewrite_mode
        self.use_fallbacks = use_fallbacks
        self.verbose = verbose
        self.stats = Phase3Stats()
        self.max_retries = max_retries
        self.enable_warmup = enable_warmup

        if use_fallbacks:
            self.model_fallbacks = [model, "qwen2.5:0.5b", "llama3.2:3b"]
        else:
            self.model_fallbacks = [model]

        if self.enable_warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Test Ollama connection with a simple query."""
        if self.verbose:
            print("[Warmup] Testing Ollama connection...", flush=True)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say OK"}],
                temperature=0.1,
                max_tokens=3,
            )
            if self.verbose:
                print(f"[Warmup] ✓ Ollama ready", flush=True)
        except Exception as e:
            if self.verbose:
                print(f"[Warmup] ✗ Warning: {type(e).__name__}: {str(e)[:100]}", flush=True)

    def _chat_with_retries(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Chat with exponential backoff retry logic."""
        attempt = 0
        last_err = None

        # Try each candidate model
        for candidate in self.model_fallbacks:
            attempt_candidate = 0
            while attempt_candidate < self.max_retries:
                attempt += 1
                attempt_candidate += 1
                try:
                    if attempt_candidate > 1:
                        wait_secs = min(2 ** (attempt_candidate - 2), 10)
                        if self.verbose:
                            print(f"  [Retry {attempt_candidate}/{self.max_retries}] Waiting {wait_secs}s...", flush=True)
                        time.sleep(wait_secs)

                    resp = self.client.chat.completions.create(
                        model=candidate,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    self.stats.api_calls += 1
                    if resp.usage:
                        self.stats.total_tokens += resp.usage.total_tokens
                    self.model = candidate
                    self.stats.synthesis_retries += attempt - 1
                    return (resp.choices[0].message.content or "").strip()

                except Exception as exc:
                    last_err = exc
                    if self.verbose:
                        print(f"  [Attempt {attempt}] {type(exc).__name__}: {str(exc)[:80]}", flush=True)

        self.stats.synthesis_failures += 1
        if self.verbose:
            print(f"  ✗ All retries exhausted: {type(last_err).__name__}", flush=True)
        return None

    @staticmethod
    def _clean_trace_text(trace: str) -> str:
        if not trace:
            return ""
        text = trace.replace("\r\n", "\n").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _truncate_text(self, text: str, max_len: int) -> str:
        """Truncate text to max_len, try to break at sentence boundary."""
        if not text or len(text) <= max_len:
            return text
        text = text[: max_len + 100]
        m = re.search(r"[\.\!\?]\s+", text[:max_len][::-1])
        if m:
            cutoff = max_len - m.start()
            return text[:cutoff].rstrip() + " [truncated]"
        return text[:max_len].rstrip() + " [truncated]"

    def _linearize_astar_trace(self, trace_text: str) -> str:
        """Extract numbered steps from A* trace."""
        lines = trace_text.split("\n")
        steps = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(?:Step\s+\d+|[-•*])\s*:?\s*(.*)", line, re.IGNORECASE)
            if m:
                steps.append(m.group(1))
            elif re.match(r"^\d+\.\s+", line):
                steps.append(re.sub(r"^\d+\.\s+", "", line))
        if not steps:
            steps = [line for line in lines if line]
        return "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))

    def _fallback_synthesize(self, cot_trace: str, astar_refined: str) -> Tuple[str, Optional[str]]:
        """Deterministic fallback: combine traces."""
        cot_clean = self._truncate_text(self._clean_trace_text(cot_trace), self.max_trace_chars // 2)
        astar_clean = self._truncate_text(self._clean_trace_text(astar_refined), self.max_trace_chars // 2)
        combined = f"{cot_clean}\n\n{astar_clean}"
        return combined, self._extract_final_answer(combined, [])

    def _extract_final_answer(self, text: str, options: List[str]) -> Optional[str]:
        """Extract answer from text."""
        if not text:
            return None
        m = re.search(r"Final\s*Answer\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            letter = re.search(r"\b([A-D])\b", raw, flags=re.IGNORECASE)
            if letter:
                return letter.group(1).upper()
            yes_no = re.search(r"\b(yes|no)\b", raw, flags=re.IGNORECASE)
            if yes_no:
                return yes_no.group(1).lower()
            return raw
        m2 = re.search(r"\banswer\s+is\s+([A-D])\b", text, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).upper()
        m3 = re.search(r"\banswer\s+is\s+(yes|no)\b", text, flags=re.IGNORECASE)
        if m3:
            return m3.group(1).lower()
        tail = "\n".join(text.splitlines()[-3:])
        letters = re.findall(r"\b([A-D])\b", tail, flags=re.IGNORECASE)
        if len(letters) == 1:
            return letters[0].upper()
        return None

    def rewrite_astar_trace(self, astar_trace: str) -> str:
        """Rewrite A* trace."""
        clean = self._clean_trace_text(astar_trace)
        if not clean:
            self.stats.rewrite_failures += 1
            return ""
        if self.rewrite_mode == "deterministic":
            return self._linearize_astar_trace(clean)
        return clean  # default

    def synthesize_reasoning(
        self,
        question: str,
        options: List[str],
        cot_trace: str,
        astar_refined_trace: str,
    ) -> Tuple[str, Optional[str], str]:
        """Synthesize CoT + A* with LLM (or fallback)."""
        options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options or []))
        cot_clean = self._truncate_text(self._clean_trace_text(cot_trace) or "N/A", self.max_trace_chars // 2)
        astar_clean = self._truncate_text(self._clean_trace_text(astar_refined_trace) or "N/A", self.max_trace_chars // 2)

        user_prompt = SYNTHESIS_USER_TEMPLATE.format(
            question=question,
            options_text=options_text or "N/A",
            cot_trace=cot_clean,
            astar_refined_trace=astar_clean,
        )

        out = self._chat_with_retries(SYNTHESIS_SYSTEM_PROMPT, user_prompt)
        if not out:
            fallback_trace, fallback_pred = self._fallback_synthesize(cot_trace, astar_refined_trace)
            return fallback_trace, fallback_pred, "fallback"

        pred = self._extract_final_answer(out, options)
        return out, pred, "llm"


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _index_results(items: List[dict]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for row in items:
        if "id" in row:
            out[int(row["id"])] = row
    return out


def run(args: argparse.Namespace) -> None:
    start = time.time()

    astar_data = _load_json(args.astar_path)
    cot_data = _load_json(args.cot_path)

    astar_results = astar_data.get("results", [])
    cot_results = cot_data.get("results", [])

    cot_by_id = _index_results(cot_results)

    runner = Phase3DualTraceRunnerV2(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_trace_chars=args.max_trace_chars,
        rewrite_mode=args.rewrite_mode,
        use_fallbacks=args.use_fallbacks,
        request_timeout=args.request_timeout,
        verbose=args.verbose,
        max_retries=args.max_retries,
        enable_warmup=args.enable_warmup,
    )

    merged_rows: List[dict] = []

    iterable = astar_results
    if args.limit and args.limit > 0:
        iterable = iterable[: args.limit]

    for arow in tqdm(iterable, desc="Phase3 v2 rewrite+synthesis"):
        qid = int(arow.get("id", -1))
        crow = cot_by_id.get(qid, {})

        context = arow.get("context") or crow.get("context") or ""
        query = arow.get("query") or crow.get("query") or ""
        options = arow.get("options") or crow.get("options") or []
        gold = str(arow.get("gold") or crow.get("gold") or "").strip()

        question_text = f"Context: {context}\nQuestion: {query}" if context else query

        astar_orig = arow.get("reasoning_trace") or arow.get("trace") or ""
        astar_refined = runner.rewrite_astar_trace(astar_orig)

        cot_trace = crow.get("reasoning_trace") or crow.get("trace") or ""

        synth_trace, pred, source = runner.synthesize_reasoning(question_text, options, cot_trace, astar_refined)

        is_correct = (pred == gold) if pred and gold else False
        if is_correct:
            runner.stats.correct += 1
        runner.stats.total += 1

        merged_rows.append(
            {
                "id": qid,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "context": context,
                "query": query,
                "options": options,
                "cot_trace": cot_trace,
                "astar_trace_original": astar_orig,
                "astar_trace_refined": astar_refined,
                "synthesized_trace": synth_trace,
                "synthesis_source": source,
            }
        )

    output = {
        "phase": "phase3_dual_trace_synthesis_v2",
        "model": runner.model,
        "base_url": args.base_url,
        "accuracy": (runner.stats.correct / runner.stats.total * 100) if runner.stats.total > 0 else 0,
        "correct": runner.stats.correct,
        "total": runner.stats.total,
        "api_calls": runner.stats.api_calls,
        "total_tokens": runner.stats.total_tokens,
        "rewrite_failures": runner.stats.rewrite_failures,
        "synthesis_failures": runner.stats.synthesis_failures,
        "synthesis_retries": runner.stats.synthesis_retries,
        "elapsed_seconds": time.time() - start,
        "results": merged_rows,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"Phase 3 v2 complete")
    print(f"Model: {runner.model}")
    print(f"Examples: {runner.stats.total}")
    print(f"Accuracy: {runner.stats.correct}/{runner.stats.total} ({output['accuracy']:.2f}%)")
    print(f"API calls: {runner.stats.api_calls}")
    print(f"Synthesis retries: {runner.stats.synthesis_retries}")
    print(f"Synthesis failures: {runner.stats.synthesis_failures}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 v2: Robust dual-trace synthesis")
    parser.add_argument("--astar-path", required=True, help="Path to A* results JSON")
    parser.add_argument("--cot-path", required=True, help="Path to CoT results JSON")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--model", default="llama3.1:8b", help="Model to use")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="Ollama base URL")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens")
    parser.add_argument("--max-trace-chars", type=int, default=1000, help="Max chars per trace (aggressive)")
    parser.add_argument("--rewrite-mode", default="deterministic", choices=["deterministic", "llm"])
    parser.add_argument("--use-fallbacks", action="store_true", help="Use fallback models")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Request timeout (sec)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    parser.add_argument("--enable-warmup", action="store_true", default=True, help="Warmup Ollama")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N items")

    args = parser.parse_args()
    run(args)
