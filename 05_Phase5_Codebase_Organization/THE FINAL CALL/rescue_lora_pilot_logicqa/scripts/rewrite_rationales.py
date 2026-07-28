#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/rescue_pairs/rescue_candidates.jsonl")
    ap.add_argument("--prompt-file", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/prompts/rewrite_prompt.txt")
    ap.add_argument("--output-jsonl", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/rescue_pairs/rewritten_rescues.jsonl")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    ap.add_argument("--max-examples", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(args.input_jsonl)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    outputs = []
    for row in tqdm(rows, desc="Rewriting rescue rationales"):
        qblock = row.get("question_block") or ""
        gold = (row.get("gold") or "").strip().upper()
        astar_trace = row.get("astar_trace") or ""

        user_prompt = prompt_template.format(
            question_block=qblock,
            gold=gold,
            astar_trace=astar_trace,
        )

        rewritten = ""
        error = None
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You rewrite reasoning traces into concise linear rationales."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            error = str(exc)

        if rewritten and not re.search(r"Final\s*Answer\s*:", rewritten, flags=re.IGNORECASE):
            rewritten = rewritten.rstrip() + f"\nFinal Answer: {gold}"

        pred = extract_final_answer(rewritten)
        out = dict(row)
        out.update(
            {
                "rewritten_rationale": rewritten,
                "rewrite_pred": pred,
                "rewrite_ok": bool(rewritten),
                "rewrite_error": error,
            }
        )
        outputs.append(out)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_jsonl(args.output_jsonl, outputs)
    print(json.dumps({
        "input": len(rows),
        "output": len(outputs),
        "ok": sum(1 for r in outputs if r.get("rewrite_ok")),
        "saved": args.output_jsonl,
    }, indent=2))


if __name__ == "__main__":
    main()
