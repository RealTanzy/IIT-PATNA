#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "results" in obj:
        return obj["results"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported JSON structure: {path}")


def norm(x):
    return (x or "").strip()


def key_of(row):
    return norm(row.get("context", "")), norm(row.get("query", ""))


def to_qblock(context, query, options):
    parts = []
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Question: {query}")
    if options:
        parts.append("Options:")
        for i, opt in enumerate(options):
            parts.append(f"{chr(65 + i)}. {opt}")
    return "\n".join(parts)


def classify_failure_category(row):
    text = f"{row.get('query','')} {row.get('context','')}".lower()
    if any(tok in text for tok in ["if ", "unless", "either", "neither", "only if", " all ", " some ", " none ", " not "]):
        return "logical_constraints"
    if any(tok in text for tok in ["which of the following", "can be derived", "inference", "conclusion"]):
        return "inference_selection"
    if len((row.get("context", "") or "").split()) > 120:
        return "long_context"
    return "other"


def split_indices(n, dev_ratio, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    if n <= 1:
        return set()
    n_dev = int(round(n * dev_ratio))
    if n >= 5:
        n_dev = max(1, n_dev)
    n_dev = min(n_dev, n - 1)
    return set(idx[:n_dev])


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--astar-file", default="/home/dibyanayan/tanzeel/Tanzeel_astar/Dibyanayan_results/logicqa_astar_100_llama3.1_8b_device0.json")
    ap.add_argument("--cot-file", default="/home/dibyanayan/tanzeel/Tanzeel_astar/Dibyanayan_results/logicqa_cot_100_llama3.1_8b_device0.json")
    ap.add_argument("--output-root", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-ratio", type=float, default=0.2)
    ap.add_argument("--preserve-multiplier", type=float, default=2.0)
    ap.add_argument("--answer-only-ratio", type=float, default=0.1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.output_root)

    astar = load_rows(args.astar_file)
    cot = load_rows(args.cot_file)

    a_map = {key_of(r): r for r in astar}
    c_map = {key_of(r): r for r in cot}
    overlap_keys = sorted(set(a_map.keys()) & set(c_map.keys()))

    paired = []
    for i, k in enumerate(overlap_keys):
        a = a_map[k]
        c = c_map[k]
        cot_correct = bool(c.get("correct"))
        astar_correct = bool(a.get("correct"))

        if cot_correct and astar_correct:
            bucket = "both_correct"
        elif cot_correct and not astar_correct:
            bucket = "cot_only"
        elif (not cot_correct) and astar_correct:
            bucket = "astar_only"
        else:
            bucket = "both_wrong"

        context = norm(a.get("context") or c.get("context"))
        query = norm(a.get("query") or c.get("query"))
        options = a.get("options") or c.get("options") or []
        gold = norm(a.get("gold") or c.get("gold")).upper()

        eid = hashlib.md5((context + "||" + query).encode("utf-8")).hexdigest()[:12]

        row = {
            "example_id": f"logicqa_{eid}",
            "context": context,
            "query": query,
            "options": options,
            "question_block": to_qblock(context, query, options),
            "gold": gold,
            "cot_pred": norm(c.get("pred", "")).upper(),
            "astar_pred": norm(a.get("pred", "")).upper(),
            "cot_correct": cot_correct,
            "astar_correct": astar_correct,
            "cot_trace": c.get("terminal_trace") or c.get("trace") or "",
            "astar_trace": a.get("terminal_trace") or a.get("trace") or "",
            "bucket": bucket,
            "failure_category": classify_failure_category({"context": context, "query": query}),
        }
        paired.append(row)

    # Split into train/dev by bucket for stable evaluation buckets
    groups = defaultdict(list)
    for r in paired:
        groups[r["bucket"]].append(r)

    for bucket, rows in groups.items():
        dev_idx = split_indices(len(rows), args.dev_ratio, rng)
        for j, r in enumerate(rows):
            r["split"] = "dev" if j in dev_idx else "train"

    paired.sort(key=lambda x: x["example_id"])

    # Training candidates
    train_rows = [r for r in paired if r["split"] == "train"]
    rescue_train = [r for r in train_rows if r["bucket"] == "astar_only"]
    preserve_pool = [r for r in train_rows if r["cot_correct"]]

    preserve_target = int(round(args.preserve_multiplier * len(rescue_train)))
    preserve_target = min(len(preserve_pool), preserve_target)
    if len(rescue_train) == 0:
        preserve_target = min(len(preserve_pool), 20)

    preserve_selected = rng.sample(preserve_pool, preserve_target) if preserve_target > 0 else []

    answer_only_count = int(round(args.answer_only_ratio * (len(rescue_train) + len(preserve_selected))))
    answer_only_count = min(answer_only_count, len(preserve_selected))
    answer_only_selected = rng.sample(preserve_selected, answer_only_count) if answer_only_count > 0 else []

    preserve_ids = {r["example_id"] for r in preserve_selected}
    answer_only_ids = {r["example_id"] for r in answer_only_selected}
    rescue_ids = {r["example_id"] for r in rescue_train}

    for r in paired:
        r["selected_rescue"] = r["example_id"] in rescue_ids
        r["selected_preserve"] = r["example_id"] in preserve_ids
        r["selected_answer_only"] = r["example_id"] in answer_only_ids

    # Write artifacts
    write_jsonl(root / "data/cot_astar_runs/logicqa_paired_all.jsonl", paired)
    write_jsonl(root / "data/rescue_pairs/rescue_candidates.jsonl", [r for r in paired if r["bucket"] == "astar_only"])
    write_jsonl(root / "data/rescue_pairs/preserve_candidates.jsonl", preserve_selected)
    write_jsonl(root / "data/rescue_pairs/answer_only_candidates.jsonl", answer_only_selected)
    write_jsonl(root / "data/eval/dev_eval_questions.jsonl", [r for r in paired if r["split"] == "dev"])

    manifest_path = root / "data/rescue_pairs/rescue_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "example_id", "bucket", "split", "gold", "cot_pred", "astar_pred",
        "cot_correct", "astar_correct", "failure_category",
        "selected_rescue", "selected_preserve", "selected_answer_only"
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in paired:
            w.writerow({k: r.get(k) for k in fields})

    # Dev bucket labels csv for analysis
    dev_csv = root / "data/eval/dev_bucket_labels.csv"
    with dev_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["example_id", "bucket", "gold", "cot_pred", "astar_pred", "cot_correct", "astar_correct", "failure_category"])
        w.writeheader()
        for r in paired:
            if r["split"] == "dev":
                w.writerow({
                    "example_id": r["example_id"],
                    "bucket": r["bucket"],
                    "gold": r["gold"],
                    "cot_pred": r["cot_pred"],
                    "astar_pred": r["astar_pred"],
                    "cot_correct": r["cot_correct"],
                    "astar_correct": r["astar_correct"],
                    "failure_category": r["failure_category"],
                })

    summary = {
        "total_overlap": len(paired),
        "bucket_counts": dict(Counter(r["bucket"] for r in paired)),
        "train_bucket_counts": dict(Counter(r["bucket"] for r in paired if r["split"] == "train")),
        "dev_bucket_counts": dict(Counter(r["bucket"] for r in paired if r["split"] == "dev")),
        "selected_rescue_train": len(rescue_train),
        "selected_preserve_train": len(preserve_selected),
        "selected_answer_only_train": len(answer_only_selected),
        "output_root": str(root),
    }
    (root / "data/rescue_pairs/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
