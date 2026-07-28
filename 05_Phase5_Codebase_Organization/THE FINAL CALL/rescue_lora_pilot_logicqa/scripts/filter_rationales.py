#!/usr/bin/env python3
import argparse
import json
import re
from difflib import SequenceMatcher
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


def extract_final_answer(text):
    if not text:
        return None
    m = re.search(r"Final\s*Answer\s*:\s*([A-D])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    tail = "\n".join(text.splitlines()[-3:])
    letters = re.findall(r"\b([A-D])\b", tail, flags=re.IGNORECASE)
    if len(letters) == 1:
        return letters[0].upper()
    return None


def count_steps(text):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    step_like = 0
    for ln in lines:
        if re.match(r"^(Step\s*\d+[:.)-]|\d+[.)-])", ln, flags=re.IGNORECASE):
            step_like += 1
    return step_like


def token_count(text):
    return len((text or "").split())


def has_search_artifacts(text):
    t = (text or "").lower()
    banned = [
        "branch", "search node", "open set", "closed set", "retry", "backtrack",
        "option 1", "option-1", "path a", "path b"
    ]
    return any(b in t for b in banned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/rescue_pairs/rewritten_rescues.jsonl")
    ap.add_argument("--output-jsonl", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/rescue_pairs/rewritten_rescues.filtered.jsonl")
    ap.add_argument("--reject-jsonl", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/rescue_pairs/rewritten_rescues.rejected.jsonl")
    ap.add_argument("--stats-json", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/rescue_pairs/filter_stats.json")
    ap.add_argument("--max-tokens", type=int, default=170)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-cot-similarity", type=float, default=0.95)
    args = ap.parse_args()

    rows = load_jsonl(args.input_jsonl)
    kept, rejected = [], []

    for r in rows:
        rationale = r.get("rewritten_rationale") or ""
        gold = (r.get("gold") or "").strip().upper()
        cot_trace = r.get("cot_trace") or ""

        reasons = []
        if not rationale.strip():
            reasons.append("empty")

        if token_count(rationale) > args.max_tokens:
            reasons.append("too_long")

        if count_steps(rationale) > args.max_steps:
            reasons.append("too_many_steps")

        if has_search_artifacts(rationale):
            reasons.append("search_artifacts")

        pred = extract_final_answer(rationale)
        if pred != gold:
            reasons.append("final_answer_mismatch")

        sim = SequenceMatcher(a=(rationale or "").lower(), b=(cot_trace or "").lower()).ratio()
        if sim >= args.max_cot_similarity:
            reasons.append("too_similar_to_wrong_cot")

        out = dict(r)
        out["filter_reasons"] = reasons
        out["rewrite_token_count"] = token_count(rationale)
        out["rewrite_step_count"] = count_steps(rationale)
        out["rewrite_vs_cot_similarity"] = sim

        if reasons:
            rejected.append(out)
        else:
            kept.append(out)

    write_jsonl(args.output_jsonl, kept)
    write_jsonl(args.reject_jsonl, rejected)

    stats = {
        "input": len(rows),
        "kept": len(kept),
        "rejected": len(rejected),
        "kept_rate": (len(kept) / len(rows) * 100.0) if rows else 0.0,
        "output_jsonl": args.output_jsonl,
        "reject_jsonl": args.reject_jsonl,
    }
    Path(args.stats_json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
