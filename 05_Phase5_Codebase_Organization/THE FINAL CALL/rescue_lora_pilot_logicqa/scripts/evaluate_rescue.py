#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_answer(text):
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


def build_user_prompt(row):
    lines = []
    if row.get("context"):
        lines.append(f"Context: {row['context']}")
    lines.append(f"Question: {row['query']}")
    opts = row.get("options") or []
    if opts:
        lines.append("Options:")
        for i, o in enumerate(opts):
            lines.append(f"{chr(65+i)}. {o}")
    lines.append("Provide a short numbered rationale and end with 'Final Answer: <A/B/C/D>'.")
    system = "You are a careful reasoning assistant. Keep reasoning concise and end with final answer."
    return f"System: {system}\n\nUser: {'\n'.join(lines)}\n\nAssistant:"


def classify_category(row):
    # Use existing label if present
    cat = row.get("failure_category")
    if cat:
        return cat

    text = f"{row.get('query','')} {row.get('context','')}".lower()
    if any(tok in text for tok in ["if ", "unless", "either", "neither", "only if", " all ", " some ", " none ", " not "]):
        return "logical_constraints"
    if any(tok in text for tok in ["which of the following", "can be derived", "inference", "conclusion"]):
        return "inference_selection"
    if len((row.get("context", "") or "").split()) > 120:
        return "long_context"
    return "other"


def load_model(base_model, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if adapter_path:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as exc:
            raise SystemExit("Failed to load adapter. Ensure peft is installed and adapter path is correct.") from exc

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, prompt, max_new_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text


def safe_rate(num, den):
    return (num / den * 100.0) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-eval-jsonl", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/eval/dev_eval_questions.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter-path", default="")
    ap.add_argument("--out-json", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/outputs/metrics/rescue_eval.json")
    ap.add_argument("--out-csv", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/outputs/metrics/sample_outputs.csv")
    args = ap.parse_args()

    rows = load_jsonl(args.dev_eval_jsonl)
    model, tokenizer = load_model(args.base_model, adapter_path=args.adapter_path or None)

    evaluated = []
    for r in tqdm(rows, desc="Evaluating rescue adapter"):
        prompt = build_user_prompt(r)
        gen = generate_answer(model, tokenizer, prompt)
        pred = extract_answer(gen)
        gold = (r.get("gold") or "").strip().upper()
        correct = bool(pred and pred == gold)

        out = dict(r)
        out["generated_text"] = gen
        out["new_pred"] = pred
        out["new_correct"] = correct
        out["failure_category"] = classify_category(r)
        evaluated.append(out)

    total = len(evaluated)
    overall_correct = sum(1 for x in evaluated if x["new_correct"])

    rescue_cases = [x for x in evaluated if x.get("bucket") == "astar_only"]
    preserve_cases = [x for x in evaluated if bool(x.get("cot_correct"))]

    rescue_correct = sum(1 for x in rescue_cases if x["new_correct"])
    preserve_correct = sum(1 for x in preserve_cases if x["new_correct"])

    # Category-wise repair inside rescue bucket
    cat_stats = defaultdict(lambda: {"total": 0, "repaired": 0})
    for x in rescue_cases:
        c = x.get("failure_category", "other")
        cat_stats[c]["total"] += 1
        if x["new_correct"]:
            cat_stats[c]["repaired"] += 1

    cat_stats = {
        k: {
            "total": v["total"],
            "repaired": v["repaired"],
            "repair_rate": safe_rate(v["repaired"], v["total"]),
        }
        for k, v in cat_stats.items()
    }

    metrics = {
        "total_dev": total,
        "overall_accuracy": safe_rate(overall_correct, total),
        "rescue_cases": len(rescue_cases),
        "rescue_rate": safe_rate(rescue_correct, len(rescue_cases)),
        "preserve_cases": len(preserve_cases),
        "preservation_rate": safe_rate(preserve_correct, len(preserve_cases)),
        "baseline_cot_accuracy_on_dev": safe_rate(sum(1 for x in evaluated if bool(x.get("cot_correct"))), total),
        "baseline_astar_accuracy_on_dev": safe_rate(sum(1 for x in evaluated if bool(x.get("astar_correct"))), total),
        "category_wise_repair": cat_stats,
        "adapter_path": args.adapter_path,
        "base_model": args.base_model,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "example_id", "bucket", "failure_category", "gold", "cot_pred", "astar_pred",
        "new_pred", "cot_correct", "astar_correct", "new_correct"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for x in evaluated:
            w.writerow({k: x.get(k) for k in fields})

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {out_json}")
    print(f"Saved per-example outputs to {out_csv}")


if __name__ == "__main__":
    main()
