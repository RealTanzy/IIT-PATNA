#!/usr/bin/env python3
"""
Batch evaluation for HotpotQA FAISS-RAG.

Outputs include per-question prediction, gold, correctness, and retrieval context.
Primary score reported as exact-match accuracy over normalized answers.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter
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

    q = row.get("query") or row.get("question") or ""
    context = row.get("context") or ""

    if context:
        return f"Context: {context}\nQuestion: {q}"
    return f"Question: {q}"


def extract_question_text(question_block: str) -> str:
    m = re.search(r"(?im)^\s*question\s*:\s*(.+?)\s*$", str(question_block or ""))
    if m:
        return normalize_answer_text(m.group(1))
    return normalize_answer_text(question_block)


def normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_answer_prefix(text: str) -> str:
    t = normalize_answer_text(text)
    t = re.sub(r"(?i)^\s*\[\s*answer\s*\]\s*", "", t)
    t = re.sub(r"(?i)^\s*final\s*answer\s*:\s*", "", t)
    t = re.sub(r"(?i)^\s*answer\s*:\s*", "", t)
    return normalize_answer_text(t)


def normalize_for_em(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    s = strip_answer_prefix(text)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_yes_no(text: str) -> Optional[str]:
    t = normalize_answer_text(strip_answer_prefix(text)).lower()
    if t in {"yes", "y", "true", "1"}:
        return "yes"
    if t in {"no", "n", "false", "0"}:
        return "no"
    yn = re.findall(r"\b(yes|no)\b", t, flags=re.IGNORECASE)
    if yn:
        return yn[-1].lower()
    return None


def is_yes_no_question(question_text: str) -> bool:
    starters = {
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "has",
        "have",
        "had",
        "am",
        "may",
        "might",
    }
    q = normalize_answer_text(question_text).lower()
    if not q:
        return False
    first = q.split()[0]
    return first in starters


def token_f1(pred_norm: str, gold_norm: str) -> float:
    pred_toks = pred_norm.split()
    gold_toks = gold_norm.split()

    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None

    final_matches = re.findall(r"(?im)^\s*final\s*answer\s*:\s*(.+?)\s*$", text)
    if final_matches:
        return strip_answer_prefix(final_matches[-1])

    answer_tag = re.findall(r"(?im)^\s*\[\s*answer\s*\]\s*(.+?)\s*$", text)
    if answer_tag:
        return strip_answer_prefix(answer_tag[-1])

    answer_colon = re.findall(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", text)
    if answer_colon:
        return strip_answer_prefix(answer_colon[-1])

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = strip_answer_prefix(lines[-1])
        # Avoid returning chain-of-thought fragments as answers.
        if re.match(r"(?i)^\s*step\s*\d+\s*:", last):
            return None
        if len(last.split()) <= 8:
            return last

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


class FaissRetriever:
    def __init__(
        self,
        kb_entries: List[dict],
        faiss_index_path: Path,
        embedding_model: str,
        hybrid_alpha: float,
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.kb_entries = kb_entries
        self.hybrid_alpha = float(hybrid_alpha)
        self.index = faiss.read_index(str(faiss_index_path))
        if self.index.ntotal != len(self.kb_entries):
            raise RuntimeError(
                f"FAISS index size ({self.index.ntotal}) does not match KB size ({len(self.kb_entries)})."
            )
        self.embedder = SentenceTransformer(embedding_model)
        self._entry_token_sets = [set(normalize_for_em(e.get("question", "")).split()) for e in self.kb_entries]

    @staticmethod
    def _lexical_overlap_score(query_tokens: set, entry_tokens: set) -> float:
        if not query_tokens or not entry_tokens:
            return 0.0
        inter = len(query_tokens & entry_tokens)
        if inter == 0:
            return 0.0
        union = len(query_tokens | entry_tokens)
        return inter / union

    def retrieve(self, question_block: str, top_k: int, exclude_example_id: Optional[str]) -> List[dict]:
        question_text = extract_question_text(question_block)
        query_tokens = set(normalize_for_em(question_text).split())

        q_emb = self.embedder.encode(
            [question_block],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)

        want = max(top_k, 1)
        search_k = min(len(self.kb_entries), max(96, want * 60))
        scores, idxs = self.index.search(q_emb, search_k)

        candidates: List[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            entry = self.kb_entries[int(idx)]
            if exclude_example_id and str(entry.get("example_id", "")) == str(exclude_example_id):
                continue

            lexical = self._lexical_overlap_score(query_tokens, self._entry_token_sets[int(idx)])
            dense = float(score)
            combined = dense + (self.hybrid_alpha * lexical)
            candidates.append(
                {
                    "score": combined,
                    "dense_score": dense,
                    "lexical_score": lexical,
                    "entry": entry,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        out = candidates[:want]
        return out


def build_prompt(target_question_block: str, retrieved: List[dict], yes_no_question: bool) -> Tuple[str, str]:
    system_msg = (
        "You are a precise reasoning assistant for HotpotQA. "
        "Use retrieved reasoning context as guidance, then answer the target question. "
        "Output only one line, exactly in this format: Final Answer: <answer>. "
        "Do not output any Step lines or extra explanation."
    )

    refs: List[str] = []
    for i, item in enumerate(retrieved, start=1):
        e = item["entry"]
        refs.append(
            f"[Reference {i} | id={e['id']} | source={e['source']} | score={item['score']:.4f} | "
            f"dense={item.get('dense_score', 0.0):.4f} | lexical={item.get('lexical_score', 0.0):.4f}]\n"
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
        "Instruction:\n"
        "Return only the final short answer span (1-6 words when possible).\n"
        "If the question is yes/no, answer only yes or no.\n"
        "Final output must be exactly one line: Final Answer: <answer>\n"
    )

    if yes_no_question:
        user_msg += "\nThis is a yes/no question. Use only yes or no."

    return system_msg, user_msg


def build_refiner_prompt(question_text: str, draft_output: str, yes_no_question: bool) -> Tuple[str, str]:
    system_msg = (
        "You are an answer extractor. "
        "Extract only the final short answer from the draft response. "
        "Output exactly one line: Final Answer: <answer>."
    )
    user_msg = (
        f"Question: {question_text}\n"
        f"Draft response:\n{draft_output}\n\n"
        "Return only the final answer text."
    )
    if yes_no_question:
        user_msg += "\nThis is yes/no. Output only yes or no."
    return system_msg, user_msg


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    artifacts = base_dir / "artifacts"

    ap = argparse.ArgumentParser(description="Evaluate HotpotQA FAISS RAG on a batch of dataset questions")
    ap.add_argument(
        "--dataset-jsonl",
        default="/home/dibyanayan/tanzeel/Hotpotqa RAg/data/cot_astar_runs/hotpotqa_paired_all.jsonl",
    )
    ap.add_argument("--split", choices=["train", "dev", "all"], default="all")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-questions", type=int, default=200)

    ap.add_argument("--kb-jsonl", default=str(artifacts / "hotpotqa_reasoning_kb.jsonl"))
    ap.add_argument("--index-meta", default=str(artifacts / "hotpotqa_index_meta.json"))
    ap.add_argument("--faiss-index", default=str(artifacts / "hotpotqa_questions.faiss"))

    ap.add_argument("--embedding-model", default="", help="Optional override. Uses index meta by default.")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument(
        "--exclude-same-example",
        dest="exclude_same_example",
        action="store_true",
        default=True,
        help="Exclude retrieval entries with the same example_id as the query row.",
    )
    ap.add_argument(
        "--allow-same-example",
        dest="exclude_same_example",
        action="store_false",
        help="Allow same example_id entries in retrieval (diagnostic upper-bound mode).",
    )
    ap.add_argument("--hybrid-alpha", type=float, default=0.35)

    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--refiner-max-new-tokens", type=int, default=20)
    ap.add_argument("--no-answer-refiner", dest="use_answer_refiner", action="store_false")
    ap.set_defaults(use_answer_refiner=True)
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument("--output-dir", default="/home/dibyanayan/tanzeel/hotpotqa rag test output")

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

    retriever = FaissRetriever(
        kb_entries=kb_entries,
        faiss_index_path=Path(args.faiss_index),
        embedding_model=embedding_model,
        hybrid_alpha=args.hybrid_alpha,
    )

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
                "hybrid_alpha": args.hybrid_alpha,
                "exclude_same_example": bool(args.exclude_same_example),
                "base_model": args.base_model,
                "use_answer_refiner": bool(args.use_answer_refiner),
                "output_dir": str(out_dir),
            },
            indent=2,
        )
    )

    t0 = time.time()
    outputs: List[dict] = []
    correct = 0
    parsed = 0
    f1_sum = 0.0

    for i, row in enumerate(selected, start=1):
        ex_id = str(row.get("example_id", ""))
        question_block = to_question_block_from_row(row)
        question_text = extract_question_text(question_block)
        yes_no_question = is_yes_no_question(question_text)

        gold_raw = strip_answer_prefix(str(row.get("gold", "")))
        gold_norm = normalize_for_em(gold_raw)

        retrieved = retriever.retrieve(
            question_block=question_block,
            top_k=args.top_k,
            exclude_example_id=ex_id if args.exclude_same_example else None,
        )

        system_msg, user_msg = build_prompt(
            target_question_block=question_block,
            retrieved=retrieved,
            yes_no_question=yes_no_question,
        )
        draft_generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            system_msg=system_msg,
            user_msg=user_msg,
            max_new_tokens=args.max_new_tokens,
        )

        pred_raw = extract_final_answer(draft_generated)

        needs_refine = (
            not pred_raw
            or len(strip_answer_prefix(pred_raw).split()) > 8
            or bool(re.match(r"(?i)^\s*step\s*\d+", str(pred_raw)))
            or (yes_no_question and normalize_yes_no(pred_raw) is None)
        )

        refined_output = ""
        if args.use_answer_refiner and needs_refine:
            r_system, r_user = build_refiner_prompt(
                question_text=question_text,
                draft_output=draft_generated,
                yes_no_question=yes_no_question,
            )
            refined_output = generate_text(
                model=model,
                tokenizer=tokenizer,
                system_msg=r_system,
                user_msg=r_user,
                max_new_tokens=args.refiner_max_new_tokens,
            )
            refined_pred = extract_final_answer(refined_output)
            if refined_pred:
                pred_raw = refined_pred

        pred_raw = strip_answer_prefix(pred_raw) if pred_raw else ""
        if yes_no_question:
            yn = normalize_yes_no(pred_raw)
            if yn is not None:
                pred_raw = yn
        pred_norm = normalize_for_em(pred_raw) if pred_raw else ""

        is_correct = bool(pred_norm) and (pred_norm == gold_norm)
        f1 = token_f1(pred_norm, gold_norm) if pred_norm else 0.0

        if pred_norm:
            parsed += 1
        if is_correct:
            correct += 1
        f1_sum += f1

        retrieved_serialized = []
        for item in retrieved:
            e = item["entry"]
            retrieved_serialized.append(
                {
                    "id": e.get("id", ""),
                    "example_id": e.get("example_id", ""),
                    "source": e.get("source", ""),
                    "score": item["score"],
                    "dense_score": item.get("dense_score", item["score"]),
                    "lexical_score": item.get("lexical_score", 0.0),
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
                "gold": gold_raw,
                "gold_normalized": gold_norm,
                "prediction": pred_raw,
                "prediction_normalized": pred_norm,
                "is_correct": is_correct,
                "f1": f1,
                "yes_no_question": yes_no_question,
                "retrieved_reasoning_path": top1_reasoning,
                "retrieved": retrieved_serialized,
                "model_output": draft_generated,
                "refined_output": refined_output,
            }
        )

        if i % 10 == 0:
            print(f"Processed {i}/{len(selected)} | parsed={parsed} | correct={correct} | avg_f1={f1_sum / i:.4f}")

    total = len(outputs)
    elapsed = time.time() - t0
    metrics = {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "avg_f1": (f1_sum / total) if total else 0.0,
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
            "hybrid_alpha": args.hybrid_alpha,
            "exclude_same_example": bool(args.exclude_same_example),
            "base_model": args.base_model,
            "max_new_tokens": args.max_new_tokens,
            "refiner_max_new_tokens": args.refiner_max_new_tokens,
            "use_answer_refiner": bool(args.use_answer_refiner),
            "output_dir": str(out_dir),
            "metrics": metrics,
        },
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions: {out_dir / 'eval_predictions.jsonl'}")
    print(f"Saved metrics: {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
