#!/usr/bin/env python3
"""
Build a paired StrategyQA JSONL (CoT + A*) for RAG pipelines.

This creates the fields expected by the StrategyQA FAISS RAG scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def load_rows(path: Path) -> List[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return obj["results"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported JSON structure in {path}")


def norm(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_yes_no(value: str) -> str:
    txt = norm(value).lower()
    if txt in {"yes", "y", "true", "1"}:
        return "yes"
    if txt in {"no", "n", "false", "0"}:
        return "no"
    return txt


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def question_of(row: dict) -> str:
    return norm(row.get("question") or row.get("query") or "")


def trace_of(row: dict, source: str) -> str:
    cand = row.get("terminal_trace") or row.get("trace") or row.get("cot_response")
    if isinstance(cand, list):
        cand = "\n".join(str(x) for x in cand)
    text = norm(cand or "")
    if text:
        return text

    # Fallback when explicit trace is unavailable.
    pred = normalize_yes_no(row.get("pred", ""))
    if source == "astar" and isinstance(row.get("nodes"), int):
        return f"Step 1: A* explored {row['nodes']} nodes and selected the best-supported answer.\nFinal Answer: {pred}"
    return f"Step 1: {source.upper()} selected the best-supported answer from available evidence.\nFinal Answer: {pred}"


def split_indices(n: int, dev_ratio: float, rng: random.Random) -> set:
    idx = list(range(n))
    rng.shuffle(idx)
    if n <= 1:
        return set()
    n_dev = int(round(n * dev_ratio))
    if n >= 5:
        n_dev = max(1, n_dev)
    n_dev = min(n_dev, n - 1)
    return set(idx[:n_dev])


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build StrategyQA paired JSONL from CoT and A* run outputs")
    ap.add_argument(
        "--astar-file",
        default="/home/dibyanayan/tanzeel/FINAL_ASTAR/qwen_results/qwen0.5b_strategyqa_astar_results.json",
    )
    ap.add_argument(
        "--cot-file",
        default="/home/dibyanayan/tanzeel/COT Reasoning qwen/qwen_results/qwen0.5b_strategyqa_cot_results.json",
    )
    ap.add_argument(
        "--out-jsonl",
        default="/home/dibyanayan/tanzeel/Strategyqa RAg/data/cot_astar_runs/strategyqa_paired_all.jsonl",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-ratio", type=float, default=0.2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    astar_rows = load_rows(Path(args.astar_file))
    cot_rows = load_rows(Path(args.cot_file))

    a_map: Dict[str, dict] = {}
    c_map: Dict[str, dict] = {}

    for r in astar_rows:
        q = question_of(r)
        if q and q not in a_map:
            a_map[q] = r

    for r in cot_rows:
        q = question_of(r)
        if q and q not in c_map:
            c_map[q] = r

    overlap = sorted(set(a_map.keys()) & set(c_map.keys()))
    paired: List[dict] = []

    for q in overlap:
        a = a_map[q]
        c = c_map[q]

        gold = normalize_yes_no(a.get("gold") or c.get("gold") or "")
        cot_pred = normalize_yes_no(c.get("pred", ""))
        astar_pred = normalize_yes_no(a.get("pred", ""))

        cot_correct = as_bool(c.get("correct", cot_pred == gold))
        astar_correct = as_bool(a.get("correct", astar_pred == gold))

        if cot_correct and astar_correct:
            bucket = "both_correct"
        elif cot_correct and not astar_correct:
            bucket = "cot_only"
        elif astar_correct and not cot_correct:
            bucket = "astar_only"
        else:
            bucket = "both_wrong"

        raw_id = c.get("id") if c.get("id") is not None else a.get("id")
        if raw_id is None or str(raw_id).strip() == "":
            raw_id = hashlib.md5(q.encode("utf-8")).hexdigest()[:12]
        example_id = f"strategyqa_{raw_id}"

        paired.append(
            {
                "example_id": example_id,
                "context": "",
                "query": q,
                "question": q,
                "options": ["yes", "no"],
                "question_block": f"Question: {q}\nOptions:\nA. yes\nB. no",
                "gold": gold,
                "cot_pred": cot_pred,
                "astar_pred": astar_pred,
                "cot_correct": cot_correct,
                "astar_correct": astar_correct,
                "cot_trace": trace_of(c, "cot"),
                "astar_trace": trace_of(a, "astar"),
                "bucket": bucket,
                "failure_category": "binary_reasoning",
            }
        )

    groups = defaultdict(list)
    for r in paired:
        groups[r["bucket"]].append(r)

    for _, rows in groups.items():
        dev_idx = split_indices(len(rows), args.dev_ratio, rng)
        for i, r in enumerate(rows):
            r["split"] = "dev" if i in dev_idx else "train"

    out_path = Path(args.out_jsonl)
    write_jsonl(out_path, paired)

    summary = {
        "astar_file": args.astar_file,
        "cot_file": args.cot_file,
        "total_astar": len(astar_rows),
        "total_cot": len(cot_rows),
        "total_overlap": len(paired),
        "bucket_counts": dict(Counter(r["bucket"] for r in paired)),
        "train_count": sum(1 for r in paired if r["split"] == "train"),
        "dev_count": sum(1 for r in paired if r["split"] == "dev"),
        "output_jsonl": str(out_path),
    }

    write_json(out_path.parent / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
