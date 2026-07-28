#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compact_trace(trace, gold, max_steps=4, max_tokens=170):
    if not trace:
        return f"Step 1: The answer follows from the constraints.\nFinal Answer: {gold}"

    lines = [ln.strip() for ln in trace.splitlines() if ln.strip()]
    clean_steps = []
    for ln in lines:
        if re.search(r"Final\s*Answer", ln, flags=re.IGNORECASE):
            continue
        ln = re.sub(r"^Step\s*\d+\s*:\s*", "", ln, flags=re.IGNORECASE)
        ln = re.sub(r"^\d+[.)-]\s*", "", ln)
        if ln:
            clean_steps.append(ln)

    clean_steps = clean_steps[:max_steps]
    out = []
    for i, s in enumerate(clean_steps, start=1):
        out.append(f"Step {i}: {s}")

    if not out:
        out = ["Step 1: Use the core constraints to determine the correct option."]

    out.append(f"Final Answer: {gold}")
    text = "\n".join(out)

    toks = text.split()
    if len(toks) > max_tokens:
        toks = toks[: max_tokens - 3] + ["...", "Final", f"Answer:{gold}"]
        text = " ".join(toks)
        if "Final Answer" not in text:
            text += f"\nFinal Answer: {gold}"

    return text


def question_block(row):
    lines = []
    ctx = (row.get("context") or "").strip()
    qry = (row.get("query") or "").strip()
    opts = row.get("options") or []
    if ctx:
        lines.append(f"Context: {ctx}")
    lines.append(f"Question: {qry}")
    if opts:
        lines.append("Options:")
        for i, o in enumerate(opts):
            lines.append(f"{chr(65 + i)}. {o}")
    lines.append("Provide a short numbered rationale and end with 'Final Answer: <A/B/C/D>'.")
    return "\n".join(lines)


def to_record(row, assistant_text, bucket):
    return {
        "messages": [
            {"role": "system", "content": "You are a careful reasoning assistant. Keep reasoning concise and end with final answer."},
            {"role": "user", "content": question_block(row)},
            {"role": "assistant", "content": assistant_text},
        ],
        "meta": {
            "example_id": row.get("example_id"),
            "bucket": bucket,
            "gold": row.get("gold"),
            "failure_category": row.get("failure_category", "other"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescue-filtered", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/rescue_pairs/rewritten_rescues.filtered.jsonl")
    ap.add_argument("--preserve-jsonl", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/rescue_pairs/preserve_candidates.jsonl")
    ap.add_argument("--answer-only-jsonl", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/rescue_pairs/answer_only_candidates.jsonl")
    ap.add_argument("--dev-eval-jsonl", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/eval/dev_eval_questions.jsonl")
    ap.add_argument("--train-out", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/lora_train/train.jsonl")
    ap.add_argument("--dev-out", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/lora_train/dev.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rescue = load_jsonl(args.rescue_filtered)
    preserve = load_jsonl(args.preserve_jsonl)
    answer_only = load_jsonl(args.answer_only_jsonl)
    dev_eval = load_jsonl(args.dev_eval_jsonl)

    rescue_by_id = {r["example_id"]: r for r in rescue}

    train_records = []

    # Preserve training examples
    for r in preserve:
        if r.get("split") != "train":
            continue
        target = compact_trace(r.get("cot_trace", ""), r.get("gold", "A"))
        train_records.append(to_record(r, target, bucket="preserve"))

    # Rescue training examples
    for rid, rr in rescue_by_id.items():
        if rr.get("split") != "train":
            continue
        target = rr.get("rewritten_rationale", "").strip()
        if not target:
            continue
        train_records.append(to_record(rr, target, bucket="rescue"))

    # Answer-only stabilization examples
    for r in answer_only:
        if r.get("split") != "train":
            continue
        target = f"Final Answer: {(r.get('gold') or '').strip().upper()}"
        train_records.append(to_record(r, target, bucket="answer_only"))

    rng.shuffle(train_records)

    # Dev supervision set for quick validation loss tracking
    dev_records = []
    for r in dev_eval:
        bucket = r.get("bucket")
        if bucket == "astar_only":
            rr = rescue_by_id.get(r.get("example_id"))
            if rr and rr.get("rewritten_rationale"):
                dev_records.append(to_record(r, rr.get("rewritten_rationale"), bucket="rescue_dev"))
        elif r.get("cot_correct"):
            target = compact_trace(r.get("cot_trace", ""), r.get("gold", "A"))
            dev_records.append(to_record(r, target, bucket="preserve_dev"))

    write_jsonl(args.train_out, train_records)
    write_jsonl(args.dev_out, dev_records)

    # Keep copy of eval questions for model-side evaluation
    eval_copy = Path(args.train_out).parent.parent / "eval" / "dev_eval_questions.jsonl"
    write_jsonl(eval_copy, dev_eval)

    summary = {
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "rescue_available": len(rescue),
        "preserve_candidates": len(preserve),
        "answer_only_candidates": len(answer_only),
        "train_out": args.train_out,
        "dev_out": args.dev_out,
        "dev_eval_copy": str(eval_copy),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
