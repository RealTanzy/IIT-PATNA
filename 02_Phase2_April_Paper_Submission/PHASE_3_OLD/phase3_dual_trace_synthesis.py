#!/usr/bin/env python3
"""Phase 3: rewrite A* traces, then synthesize CoT + refined A* reasoning."""

import argparse
import json
import os
import re
import time
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


class Phase3DualTraceRunner:
    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        max_trace_chars: int = 3500,
        rewrite_mode: str = "deterministic",
        use_fallbacks: bool = False,
        request_timeout: float = 180.0,
        verbose: bool = False,
    ):
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=request_timeout, max_retries=1)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_trace_chars = max_trace_chars
        self.rewrite_mode = rewrite_mode
        self.use_fallbacks = use_fallbacks
        self.verbose = verbose
        self.stats = Phase3Stats()

        if use_fallbacks:
            self.model_fallbacks = [
                model,
                "llama3.1:8b",
                "qwen3-coder-next:latest",
                "llama3.2:3b",
                "qwen2.5:0.5b",
            ]
        else:
            self.model_fallbacks = [model]

    def _chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        last_err = None
        for candidate in self.model_fallbacks:
            try:
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
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_err = exc
                continue

        if self.verbose and last_err is not None:
            print(f"chat failed after fallbacks: {last_err}")
        return None

    @staticmethod
    def _clean_trace_text(trace: str) -> str:
        if not trace:
            return ""
        text = trace.replace("\r\n", "\n").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        keep = max(500, max_chars - 60)
        return text[:keep].rstrip() + "\n... [truncated]"

    @staticmethod
    def _linearize_astar_trace(trace: str) -> str:
        """Convert possibly noisy A* trace to a strict linear step list without changing claims."""
        clean = trace.replace("\r\n", "\n").strip()
        if not clean:
            return ""

        # Prefer explicit numbered steps as-is.
        step_matches = re.findall(
            r"(?:^|\n)\s*Step\s*\d+\s*:\s*(.+?)(?=(?:\n\s*Step\s*\d+\s*:)|$)",
            clean,
            flags=re.IGNORECASE | re.DOTALL,
        )
        steps = [re.sub(r"\s+", " ", s).strip() for s in step_matches if s.strip()]

        # Fallback: line-based split for traces missing explicit step tags.
        if not steps:
            raw_lines = [ln.strip(" -\t") for ln in clean.split("\n") if ln.strip()]
            steps = [re.sub(r"\s+", " ", ln).strip() for ln in raw_lines if ln]

        # Keep output strictly linear and minimal.
        linear = [f"Step {i+1}: {content}" for i, content in enumerate(steps)]

        final = re.search(r"Final\s*Answer\s*:\s*([^\n]+)", clean, flags=re.IGNORECASE)
        if final:
            linear.append(f"Final Answer: {final.group(1).strip()}")

        return "\n".join(linear).strip()

    @staticmethod
    def _extract_final_answer(text: str, options: Optional[List[str]] = None) -> Optional[str]:
        if not text:
            return None

        # Prefer explicit final answer marker.
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

        # Fallback for option letters.
        m2 = re.search(r"\banswer\s+is\s+([A-D])\b", text, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).upper()

        # Fallback for yes/no tasks.
        m3 = re.search(r"\banswer\s+is\s+(yes|no)\b", text, flags=re.IGNORECASE)
        if m3:
            return m3.group(1).lower()

        # Last fallback: select a unique option letter in last lines.
        tail = "\n".join(text.splitlines()[-3:])
        letters = re.findall(r"\b([A-D])\b", tail, flags=re.IGNORECASE)
        if len(letters) == 1:
            return letters[0].upper()

        return None

    def rewrite_astar_trace(self, astar_trace: str) -> str:
        clean = self._clean_trace_text(astar_trace)
        if not clean:
            self.stats.rewrite_failures += 1
            return ""

        if self.rewrite_mode == "deterministic":
            return self._linearize_astar_trace(clean)

        user_prompt = REWRITE_USER_TEMPLATE.format(astar_trace=clean)
        out = self._chat(REWRITE_SYSTEM_PROMPT, user_prompt)
        if not out:
            self.stats.rewrite_failures += 1
            return clean
        return out

    def _fallback_synthesize(self, cot_trace: str, astar_refined_trace: str) -> Tuple[str, Optional[str]]:
        """Deterministic backup when LLM synthesis is unavailable."""
        base = self._clean_trace_text(astar_refined_trace) or self._clean_trace_text(cot_trace)
        base = self._truncate_text(base, self.max_trace_chars)

        pred = self._extract_final_answer(cot_trace) or self._extract_final_answer(astar_refined_trace)

        if base and pred and not re.search(r"Final\s*Answer\s*:", base, flags=re.IGNORECASE):
            base = f"{base}\nFinal Answer: {pred}"

        if not base:
            base = "Step 1: Unable to synthesize due model unavailability.\nFinal Answer:"

        return base, pred

    def synthesize_reasoning(
        self,
        question: str,
        options: List[str],
        cot_trace: str,
        astar_refined_trace: str,
    ) -> Tuple[str, Optional[str], str]:
        options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options or []))
        cot_clean = self._truncate_text(self._clean_trace_text(cot_trace) or "N/A", self.max_trace_chars)
        astar_clean = self._truncate_text(self._clean_trace_text(astar_refined_trace) or "N/A", self.max_trace_chars)
        user_prompt = SYNTHESIS_USER_TEMPLATE.format(
            question=question,
            options_text=options_text or "N/A",
            cot_trace=cot_clean,
            astar_refined_trace=astar_clean,
        )
        out = self._chat(SYNTHESIS_SYSTEM_PROMPT, user_prompt)
        if not out:
            self.stats.synthesis_failures += 1
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

    runner = Phase3DualTraceRunner(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_trace_chars=args.max_trace_chars,
        rewrite_mode=args.rewrite_mode,
        use_fallbacks=args.use_fallbacks,
        request_timeout=args.request_timeout,
        verbose=args.verbose,
    )

    merged_rows: List[dict] = []

    iterable = astar_results
    if args.limit and args.limit > 0:
        iterable = iterable[: args.limit]

    for arow in tqdm(iterable, desc="Phase3 rewrite+synthesis"):
        qid = int(arow.get("id", -1))
        crow = cot_by_id.get(qid, {})

        context = arow.get("context") or crow.get("context") or ""
        query = arow.get("query") or crow.get("query") or ""
        options = arow.get("options") or crow.get("options") or []
        gold = str(arow.get("gold") or crow.get("gold") or "").strip()

        question_text = f"Context: {context}\nQuestion: {query}" if context else query

        astar_trace = str(arow.get("trace") or arow.get("terminal_trace") or "")
        cot_trace = str(crow.get("trace") or crow.get("cot_response") or crow.get("terminal_trace") or "")

        refined_astar = runner.rewrite_astar_trace(astar_trace)
        synthesized_trace, pred, synthesis_source = runner.synthesize_reasoning(
            question_text,
            options,
            cot_trace,
            refined_astar,
        )

        pred_norm = (pred or "").strip()
        correct = bool(pred_norm and gold and pred_norm.upper() == gold.upper())

        runner.stats.total += 1
        if correct:
            runner.stats.correct += 1

        merged_rows.append(
            {
                "id": qid,
                "gold": gold,
                "pred": pred_norm,
                "correct": correct,
                "context": context,
                "query": query,
                "options": options,
                "cot_trace": cot_trace,
                "astar_trace_original": astar_trace,
                "astar_trace_refined": refined_astar,
                "synthesized_trace": synthesized_trace,
                "synthesis_source": synthesis_source,
            }
        )

    acc = (runner.stats.correct / runner.stats.total * 100.0) if runner.stats.total else 0.0
    output = {
        "phase": "phase3_dual_trace_synthesis",
        "model": runner.model,
        "base_url": args.base_url,
        "accuracy": acc,
        "correct": runner.stats.correct,
        "total": runner.stats.total,
        "api_calls": runner.stats.api_calls,
        "total_tokens": runner.stats.total_tokens,
        "rewrite_failures": runner.stats.rewrite_failures,
        "synthesis_failures": runner.stats.synthesis_failures,
        "elapsed_seconds": time.time() - start,
        "results": merged_rows,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Phase 3 complete")
    print(f"Model: {runner.model}")
    print(f"Examples: {runner.stats.total}")
    print(f"Accuracy: {runner.stats.correct}/{runner.stats.total} ({acc:.2f}%)")
    print(f"API calls: {runner.stats.api_calls}")
    print(f"Rewrite failures: {runner.stats.rewrite_failures}")
    print(f"Synthesis failures: {runner.stats.synthesis_failures}")
    print(f"Output: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 3: refine A* trace then synthesize with CoT")
    p.add_argument("--astar-path", required=True, help="Path to A* results JSON")
    p.add_argument("--cot-path", required=True, help="Path to CoT results JSON")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--model", default=os.getenv("PHASE3_MODEL", "llama3.1:8b"))
    p.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=450)
    p.add_argument("--max-trace-chars", type=int, default=3500)
    p.add_argument("--rewrite-mode", choices=["deterministic", "llm"], default="deterministic")
    p.add_argument("--use-fallbacks", action="store_true")
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
