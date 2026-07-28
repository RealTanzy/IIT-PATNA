#!/usr/bin/env python3
"""
Batch evaluation for LogicQA FAISS RAG.

Outputs include, per question:
- question asked
- retrieved reasoning path/context
- model reasoning output
- predicted answer
- gold answer
- correctness flag
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_filter(rows: List[dict], split_name: str) -> List[dict]:
    split_name = (split_name or "all").lower()
    if split_name == "all":
        return rows
    return [r for r in rows if str(r.get("split", "")).lower() == split_name]


def to_question_block_from_row(row: dict) -> str:
    if row.get("question_block"):
        return str(row["question_block"])

    parts: List[str] = []
    if row.get("context"):
        parts.append(f"Context: {row['context']}")
    parts.append(f"Question: {row.get('query', '')}")

    options = row.get("options") or []
    if options:
        parts.append("Options:")
        for i, opt in enumerate(options):
            parts.append(f"{chr(65 + i)}. {opt}")

    return "\n".join(parts)


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None

    m = re.search(r"Final\s*Answer\s*:\s*([A-D])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m2 = re.findall(r"(?:the\s+)?answer\s*(?:is|:)\s*([A-D])\b", text, flags=re.IGNORECASE)
    if m2:
        return m2[-1].upper()

    m3 = re.findall(r"\boption\s*([A-D])\b", text, flags=re.IGNORECASE)
    if m3:
        return m3[-1].upper()

    tail = "\n".join(text.splitlines()[-4:])
    letters = re.findall(r"\b([A-D])\b", tail, flags=re.IGNORECASE)
    if len(letters) == 1:
        return letters[0].upper()
    return None


def load_generator(base_model: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_msg: str,
    user_msg: str,
    max_new_tokens: int,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def retrieve(
    question_block: str,
    kb_entries: List[dict],
    faiss_index_path: Path,
    embedding_model: str,
    top_k: int,
    exclude_example_id: Optional[str],
) -> List[dict]:
    import faiss
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(faiss_index_path))
    if index.ntotal != len(kb_entries):
        raise RuntimeError(
            f"FAISS index size ({index.ntotal}) does not match KB size ({len(kb_entries)})."
        )

    model = SentenceTransformer(embedding_model)
    q_emb = model.encode(
        [question_block],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q_emb = np.asarray(q_emb, dtype=np.float32)

    want = max(top_k, 1)
    search_k = min(len(kb_entries), max(32, want * 20))
    scores, idxs = index.search(q_emb, search_k)

    out: List[dict] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        entry = kb_entries[int(idx)]
        if exclude_example_id and str(entry.get("example_id", "")) == str(exclude_example_id):
            continue
        out.append({"score": float(score), "entry": entry})
        if len(out) >= want:
            break
    return out


def build_prompt(target_question_block: str, retrieved: List[dict]) -> Tuple[str, str]:
    system_msg = (
        "You are a careful multiple-choice reasoner for LogicQA. "
        "Use the retrieved reasoning context as guidance, then solve the target question. "
        "Keep reasoning short. End with exactly: Final Answer: <A/B/C/D>."
    )

    refs: List[str] = []
    for i, item in enumerate(retrieved, start=1):
        e = item["entry"]
        refs.append(
            f"[Reference {i} | id={e['id']} | source={e['source']} | cosine={item['score']:.4f}]\n"
            f"Question:\n{e['question']}\n\n"
            f"Reasoning Context:\n{e['reasoning']}\n\n"
            f"Answer: {e['answer']}"
        )

    refs_block = "\n\n".join(refs) if refs else "(No references retrieved.)"

    user_msg = (
        "Retrieved examples:\n"
        f"{refs_block}\n\n"
        "Target question:\n"
        f"{target_question_block}\n\n"
        "Output format:\n"
        "Step 1: <short reasoning>\n"
        "Step 2: <short reasoning or omit if not needed>\n"
        "Final Answer: <A/B/C/D>"
    )
    return system_msg, user_msg


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    artifacts = base_dir / "artifacts"

    ap = argparse.ArgumentParser(description="Evaluate LogicQA FAISS RAG on a batch of dataset questions")
    ap.add_argument(
        "--dataset-jsonl",
        default="/home/dibyanayan/tanzeel/THE RAG APPROACH/logicqa_651_pair/data/cot_astar_runs/logicqa_paired_all.jsonl",
    )
    ap.add_argument("--split", choices=["train", "dev", "all"], default="all")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-questions", type=int, default=200)

    ap.add_argument("--kb-jsonl", default=str(artifacts / "logicqa_reasoning_kb.jsonl"))
    ap.add_argument("--index-meta", default=str(artifacts / "logicqa_index_meta.json"))
    ap.add_argument("--faiss-index", default=str(artifacts / "logicqa_questions.faiss"))

    ap.add_argument("--embedding-model", default="", help="Optional override. Uses index meta by default.")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--exclude-same-example", action="store_true", default=True)

    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument("--output-dir", default="/home/dibyanayan/tanzeel/logicqa test output")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = load_jsonl(Path(args.dataset_jsonl))
    dataset_rows = split_filter(dataset_rows, args.split)

    if args.start_index < 0 or args.start_index >= len(dataset_rows):
        raise SystemExit(f"start-index {args.start_index} out of range for {len(dataset_rows)} rows")

    selected = dataset_rows[args.start_index : args.start_index + max(args.max_questions, 0)]
    if not selected:
        raise SystemExit("No evaluation rows selected.")

    meta = load_json(Path(args.index_meta))
    kb_entries = load_jsonl(Path(args.kb_jsonl))

    embedding_model = args.embedding_model.strip() or str(meta.get("embedding_model", "")).strip()
    if not embedding_model:
        raise SystemExit("Could not determine embedding model. Provide --embedding-model or valid index meta.")

    model, tokenizer = load_generator(
        base_model=args.base_model,
        trust_remote_code=bool(args.trust_remote_code),
    )

    print(
        json.dumps(
            {
                "dataset_jsonl": args.dataset_jsonl,
                "selected_rows": len(selected),
                "split": args.split,
                "start_index": args.start_index,
                "kb_size": len(kb_entries),
                "top_k": args.top_k,
                "exclude_same_example": bool(args.exclude_same_example),
                "base_model": args.base_model,
                "output_dir": str(out_dir),
            },
            indent=2,
        )
    )

    t0 = time.time()
    outputs: List[dict] = []
    correct = 0
    parsed = 0

    for i, row in enumerate(selected, start=1):
        ex_id = str(row.get("example_id", ""))
        gold = str(row.get("gold", "")).strip().upper()
        question_block = to_question_block_from_row(row)

        retrieved = retrieve(
            question_block=question_block,
            kb_entries=kb_entries,
            faiss_index_path=Path(args.faiss_index),
            embedding_model=embedding_model,
            top_k=args.top_k,
            exclude_example_id=ex_id if args.exclude_same_example else None,
        )

        system_msg, user_msg = build_prompt(target_question_block=question_block, retrieved=retrieved)
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            system_msg=system_msg,
            user_msg=user_msg,
            max_new_tokens=args.max_new_tokens,
        )
        pred = extract_final_answer(generated)

        is_correct = bool(pred) and (pred == gold)
        if pred:
            parsed += 1
        if is_correct:
            correct += 1

        retrieved_serialized = []
        for item in retrieved:
            e = item["entry"]
            retrieved_serialized.append(
                {
                    "id": e.get("id", ""),
                    "example_id": e.get("example_id", ""),
                    "source": e.get("source", ""),
                    "score": item["score"],
                    "question": e.get("question", ""),
                    "reasoning": e.get("reasoning", ""),
                    "answer": e.get("answer", ""),
                }
            )

        top1_reasoning = retrieved_serialized[0]["reasoning"] if retrieved_serialized else ""

        outputs.append(
            {
                "index": i,
                "example_id": ex_id,
                "question": question_block,
                "gold": gold,
                "prediction": pred,
                "is_correct": is_correct,
                "retrieved_reasoning_path": top1_reasoning,
                "retrieved": retrieved_serialized,
                "model_output": generated,
            }
        )

        if i % 10 == 0:
            print(f"Processed {i}/{len(selected)} | parsed={parsed} | correct={correct}")

    total = len(outputs)
    elapsed = time.time() - t0
    metrics = {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "parsed_rate": (parsed / total) if total else 0.0,
        "elapsed_sec": elapsed,
        "avg_sec_per_question": (elapsed / total) if total else 0.0,
    }

    write_jsonl(out_dir / "eval_predictions.jsonl", outputs)
    write_json(out_dir / "eval_metrics.json", metrics)
    write_json(
        out_dir / "run_config.json",
        {
            "dataset_jsonl": args.dataset_jsonl,
            "split": args.split,
            "start_index": args.start_index,
            "max_questions": args.max_questions,
            "kb_jsonl": args.kb_jsonl,
            "index_meta": args.index_meta,
            "faiss_index": args.faiss_index,
            "embedding_model": embedding_model,
            "top_k": args.top_k,
            "exclude_same_example": bool(args.exclude_same_example),
            "base_model": args.base_model,
            "max_new_tokens": args.max_new_tokens,
            "output_dir": str(out_dir),
            "metrics": metrics,
        },
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions: {out_dir / 'eval_predictions.jsonl'}")
    print(f"Saved metrics: {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
