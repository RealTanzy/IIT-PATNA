#!/usr/bin/env python3
"""
Build a HotpotQA reasoning KB and a FAISS index for retrieval.

Core behavior:
- Keep CoT traces where CoT is correct.
- Keep A* traces where A* is correct.
- Convert traces into compact linear reasoning.
- Build FAISS cosine index over question text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_answer_text(text: str) -> str:
    t = normalize_space(text)
    t = re.sub(r"(?i)^\s*\[\s*answer\s*\]\s*", "", t)
    t = re.sub(r"(?i)^\s*final\s*answer\s*:\s*", "", t)
    return t


def to_question_block(row: dict) -> str:
    if row.get("question_block"):
        return str(row["question_block"])

    q = row.get("query") or row.get("question") or ""
    context = row.get("context") or ""

    if context:
        return f"Context: {context}\nQuestion: {q}"
    return f"Question: {q}"


def _split_trace_units(trace: str) -> List[str]:
    trace = str(trace or "").replace("\r", "\n")

    lines = [
        ln
        for ln in trace.split("\n")
        if not re.search(r"final\s*answer\s*:", ln, flags=re.IGNORECASE)
    ]
    body = "\n".join(lines)

    step_parts = re.split(r"(?i)\bstep\s*\d+\s*:\s*", body)
    step_parts = [normalize_space(p) for p in step_parts if normalize_space(p)]
    if len(step_parts) > 1:
        return step_parts

    rough_lines = [normalize_space(x) for x in body.split("\n") if normalize_space(x)]
    if not rough_lines:
        return []

    units: List[str] = []
    for line in rough_lines:
        sent_parts = re.split(r"(?<=[.!?])\s+", line)
        for p in sent_parts:
            p = normalize_space(re.sub(r"^\d+[\)\.\-:\s]+", "", p))
            if p:
                units.append(p)
    return units


def linearize_reasoning(trace: str, final_answer: str, max_steps: int, max_step_chars: int) -> str:
    units = _split_trace_units(trace)

    if max_steps > 0:
        units = units[:max_steps]

    out_lines: List[str] = []
    for i, unit in enumerate(units, start=1):
        if max_step_chars > 0 and len(unit) > max_step_chars:
            unit = unit[: max_step_chars - 3].rstrip() + "..."
        out_lines.append(f"Step {i}: {unit}")

    if final_answer:
        out_lines.append(f"Final Answer: {final_answer}")

    return "\n".join(out_lines)


def split_filter(rows: List[dict], split_name: str) -> List[dict]:
    split_name = (split_name or "all").lower()
    if split_name == "all":
        return rows
    return [r for r in rows if str(r.get("split", "")).lower() == split_name]


def fallback_trace(row: dict, source_name: str) -> str:
    candidate = row.get("terminal_trace") or row.get("trace") or row.get("cot_response") or row.get("reasoning") or ""
    if isinstance(candidate, list):
        candidate = "\n".join(str(x) for x in candidate)
    txt = str(candidate or "").strip()
    if txt:
        return txt

    pred = normalize_answer_text(row.get("pred", ""))
    if source_name.lower().startswith("a*"):
        nodes = row.get("nodes")
        if isinstance(nodes, int):
            return f"Step 1: A* explored {nodes} nodes and selected the best-supported path.\nFinal Answer: {pred}"
    return f"Step 1: {source_name} selected the best-supported answer.\nFinal Answer: {pred}"


def build_reasoning_entries(
    rows: List[dict],
    split_name: str,
    max_steps: int,
    max_step_chars: int,
    cot_min_f1: float,
    astar_min_f1: float,
) -> Tuple[List[dict], Dict[str, int], int]:
    filtered = split_filter(rows, split_name)
    counts = Counter()
    entries: List[dict] = []

    for row in filtered:
        example_id = str(row.get("example_id", "")).strip()
        question = to_question_block(row)
        answer = normalize_answer_text(row.get("gold", ""))
        cot_f1 = as_float(row.get("cot_f1", 0.0), default=0.0)
        astar_f1 = as_float(row.get("astar_f1", 0.0), default=0.0)

        keep_cot = as_bool(row.get("cot_correct", False)) or (cot_f1 >= cot_min_f1)
        keep_astar = as_bool(row.get("astar_correct", False)) or (astar_f1 >= astar_min_f1)

        if keep_cot:
            trace = str(row.get("cot_trace") or fallback_trace(row, "CoT"))
            reasoning = linearize_reasoning(
                trace=trace,
                final_answer=answer,
                max_steps=max_steps,
                max_step_chars=max_step_chars,
            )
            entries.append(
                {
                    "id": f"{example_id}::cot",
                    "question": question,
                    "reasoning": reasoning,
                    "answer": answer,
                    "source": "cot",
                    "example_id": example_id,
                    "source_f1": cot_f1,
                    "source_exact_correct": as_bool(row.get("cot_correct", False)),
                }
            )
            counts["cot"] += 1
            if not as_bool(row.get("cot_correct", False)):
                counts["cot_soft_f1"] += 1

        if keep_astar:
            trace = str(row.get("astar_trace") or fallback_trace(row, "A*"))
            reasoning = linearize_reasoning(
                trace=trace,
                final_answer=answer,
                max_steps=max_steps,
                max_step_chars=max_step_chars,
            )
            entries.append(
                {
                    "id": f"{example_id}::astar",
                    "question": question,
                    "reasoning": reasoning,
                    "answer": answer,
                    "source": "astar",
                    "example_id": example_id,
                    "source_f1": astar_f1,
                    "source_exact_correct": as_bool(row.get("astar_correct", False)),
                }
            )
            counts["astar"] += 1
            if not as_bool(row.get("astar_correct", False)):
                counts["astar_soft_f1"] += 1

    return entries, dict(counts), len(filtered)


def build_faiss_index(entries: List[dict], embedding_model: str):
    import faiss
    from sentence_transformers import SentenceTransformer

    texts = [e["question"] for e in entries]
    model = SentenceTransformer(embedding_model)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings.shape[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build HotpotQA CoT+A* correct-reasoning KB with FAISS index")
    ap.add_argument(
        "--paired-jsonl",
        default="/home/dibyanayan/tanzeel/Hotpotqa RAg/data/cot_astar_runs/hotpotqa_paired_all.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="/home/dibyanayan/tanzeel/Hotpotqa RAg/artifacts",
    )
    ap.add_argument("--split", choices=["train", "dev", "all"], default="all")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-step-chars", type=int, default=220)
    ap.add_argument(
        "--cot-min-f1",
        type=float,
        default=1.01,
        help="Include CoT entries when cot_f1 >= threshold, even if not exact-correct. Set >1 to disable.",
    )
    ap.add_argument(
        "--astar-min-f1",
        type=float,
        default=1.01,
        help="Include A* entries when astar_f1 >= threshold, even if not exact-correct. Set >1 to disable.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    paired_path = Path(args.paired_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(paired_path)
    entries, counts, filtered_count = build_reasoning_entries(
        rows=rows,
        split_name=args.split,
        max_steps=args.max_steps,
        max_step_chars=args.max_step_chars,
        cot_min_f1=args.cot_min_f1,
        astar_min_f1=args.astar_min_f1,
    )

    if not entries:
        raise SystemExit("No KB entries generated. Check split and source data.")

    cot_entries = [e for e in entries if e["source"] == "cot"]
    astar_entries = [e for e in entries if e["source"] == "astar"]

    kb_all_path = out_dir / "hotpotqa_reasoning_kb.jsonl"
    kb_cot_path = out_dir / "hotpotqa_reasoning_kb_cot.jsonl"
    kb_astar_path = out_dir / "hotpotqa_reasoning_kb_astar.jsonl"

    write_jsonl(kb_all_path, entries)
    write_jsonl(kb_cot_path, cot_entries)
    write_jsonl(kb_astar_path, astar_entries)

    index, dim = build_faiss_index(entries=entries, embedding_model=args.embedding_model)

    import faiss

    faiss_path = out_dir / "hotpotqa_questions.faiss"
    faiss.write_index(index, str(faiss_path))

    meta = {
        "paired_jsonl": str(paired_path),
        "split": args.split,
        "embedding_model": args.embedding_model,
        "cot_min_f1": args.cot_min_f1,
        "astar_min_f1": args.astar_min_f1,
        "faiss_metric": "inner_product_on_l2_normalized_vectors",
        "dimension": dim,
        "kb_size": len(entries),
        "cot_size": len(cot_entries),
        "astar_size": len(astar_entries),
        "filtered_rows": filtered_count,
        "all_rows": len(rows),
        "kb_jsonl": str(kb_all_path),
        "kb_cot_jsonl": str(kb_cot_path),
        "kb_astar_jsonl": str(kb_astar_path),
        "faiss_index": str(faiss_path),
    }
    write_json(out_dir / "hotpotqa_index_meta.json", meta)

    write_json(
        out_dir / "kb_summary.json",
        {
            "kb_size": len(entries),
            "source_breakdown": counts,
            "split": args.split,
            "filtered_rows": filtered_count,
            "all_rows": len(rows),
        },
    )

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
