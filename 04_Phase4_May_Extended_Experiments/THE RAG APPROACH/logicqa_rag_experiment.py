#!/usr/bin/env python3
"""
LogicQA CoT+A* Complementary RAG Experiment (Llama 3.1 8B)

Pipeline:
1) Build complementary knowledge base from paired CoT/A* runs.
2) Convert reasoning traces into compact linearized context.
3) Retrieve top-k similar examples with cosine similarity.
4) Generate answer with Llama 3.1 8B using retrieved context.
5) Report accuracy and per-bucket metrics.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def to_question_block(row: dict) -> str:
    if row.get("question_block"):
        return str(row["question_block"])

    parts: List[str] = []
    if row.get("context"):
        parts.append(f"Context: {row['context']}")
    parts.append(f"Question: {row.get('query', '')}")

    options = row.get("options") or []
    if options:
        parts.append("Options:")
        for idx, opt in enumerate(options):
            parts.append(f"{chr(65 + idx)}. {opt}")

    return "\n".join(parts)


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None

    m = re.search(r"Final\s*Answer\s*:\s*([A-D])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Common fallback forms from instruction-tuned models.
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


def _split_trace_units(trace: str) -> List[str]:
    trace = trace or ""
    trace = trace.replace("\r", "\n")

    # Remove explicit final-answer lines from body; final answer is re-appended later.
    lines = [ln for ln in trace.split("\n") if not re.search(r"final\s*answer\s*:", ln, flags=re.IGNORECASE)]
    body = "\n".join(lines)

    # First split by explicit "Step n:" markers.
    step_parts = re.split(r"(?i)\bstep\s*\d+\s*:\s*", body)
    step_parts = [normalize_space(p) for p in step_parts if normalize_space(p)]

    if len(step_parts) > 1:
        return step_parts

    # Fallback: split by line then by sentence punctuation.
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


def linearize_reasoning(
    trace: str,
    final_answer: str,
    max_steps: int = 8,
    max_step_chars: int = 220,
) -> str:
    units = _split_trace_units(trace)

    # Keep unique reasoning units in original order.
    seen = set()
    kept: List[str] = []
    for u in units:
        cleaned = normalize_space(u)
        if not cleaned:
            continue
        key = re.sub(r"[^a-z0-9]+", "", cleaned.lower())
        if len(key) < 12:
            continue
        if key in seen:
            continue
        seen.add(key)
        if len(cleaned) > max_step_chars:
            clipped = cleaned[:max_step_chars]
            if " " in clipped:
                clipped = clipped.rsplit(" ", 1)[0]
            cleaned = clipped + "..."
        kept.append(cleaned)
        if len(kept) >= max_steps:
            break

    if not kept:
        kept = ["Identify the key claim, test options against the context, and choose the best-supported option."]

    out_lines = [f"Step {i + 1}: {step}" for i, step in enumerate(kept)]
    out_lines.append(f"Final Answer: {final_answer}")
    return "\n".join(out_lines)


def bucket_of(row: dict) -> str:
    if row.get("bucket"):
        return str(row["bucket"])
    cot_correct = bool(row.get("cot_correct"))
    astar_correct = bool(row.get("astar_correct"))
    if cot_correct and astar_correct:
        return "both_correct"
    if cot_correct and not astar_correct:
        return "cot_only"
    if astar_correct and not cot_correct:
        return "astar_only"
    return "both_wrong"


def split_filter(rows: List[dict], split_name: str) -> List[dict]:
    split_name = (split_name or "all").lower()
    if split_name == "all":
        return rows
    return [r for r in rows if str(r.get("split", "")).lower() == split_name]


@dataclass
class KBEntry:
    id: str
    question: str
    context: str
    style: str
    source_bucket: str
    answer: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "style": self.style,
            "source_bucket": self.source_bucket,
            "answer": self.answer,
        }


def build_complementary_kb(
    rows: List[dict],
    kb_split: str,
    max_steps: int,
    max_step_chars: int,
) -> Tuple[List[KBEntry], Dict[str, int]]:
    filtered = split_filter(rows, kb_split)
    entries: List[KBEntry] = []

    stats = Counter()
    for r in filtered:
        bucket = bucket_of(r)
        if bucket not in {"cot_only", "astar_only"}:
            continue

        ex_id = str(r.get("example_id", ""))
        question = to_question_block(r)
        gold = str(r.get("gold", "")).strip().upper()

        if bucket == "cot_only":
            trace = str(r.get("cot_trace", ""))
            style = "cot_preserved"
        else:
            trace = str(r.get("astar_trace", ""))
            style = "astar_linearized"

        reasoning = linearize_reasoning(
            trace=trace,
            final_answer=gold,
            max_steps=max_steps,
            max_step_chars=max_step_chars,
        )

        entries.append(
            KBEntry(
                id=ex_id,
                question=question,
                context=reasoning,
                style=style,
                source_bucket=bucket,
                answer=gold,
            )
        )
        stats[bucket] += 1

    return entries, dict(stats)


class CosineRetriever:
    def __init__(self, method: str, embedding_model: str):
        self.method = method
        self.embedding_model = embedding_model
        self.ids: List[str] = []
        self.entries: List[KBEntry] = []
        self._fallback_reason = ""

        # sbert members
        self._sbert_model = None
        self._sbert_matrix = None

        # tfidf members
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

    @property
    def actual_method(self) -> str:
        if self._sbert_model is not None:
            return "sbert"
        if self._tfidf_vectorizer is not None:
            return "tfidf"
        return "unknown"

    @property
    def fallback_reason(self) -> str:
        return self._fallback_reason

    def fit(self, entries: List[KBEntry]) -> None:
        self.entries = entries
        self.ids = [e.id for e in entries]
        texts = [e.question for e in entries]

        requested = self.method.lower()
        use_sbert = requested in {"auto", "sbert"}

        if use_sbert:
            try:
                from sentence_transformers import SentenceTransformer

                self._sbert_model = SentenceTransformer(self.embedding_model)
                emb = self._sbert_model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=64,
                )
                self._sbert_matrix = emb.astype(np.float32)
                return
            except Exception as exc:  # pragma: no cover - best effort fallback
                self._sbert_model = None
                self._sbert_matrix = None
                self._fallback_reason = f"SBERT unavailable ({type(exc).__name__}), switched to TF-IDF"
                if requested == "sbert":
                    # Hard fail only when user explicitly requests SBERT.
                    raise

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            lowercase=True,
        )
        mat = self._tfidf_vectorizer.fit_transform(texts)
        self._tfidf_matrix = normalize(mat)

    def query(self, text: str, top_k: int, exclude_id: Optional[str] = None) -> List[dict]:
        if not self.entries:
            return []

        if self._sbert_model is not None:
            q_emb = self._sbert_model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].astype(np.float32)
            scores = self._sbert_matrix @ q_emb
        else:
            from sklearn.preprocessing import normalize

            q = self._tfidf_vectorizer.transform([text])
            q = normalize(q)
            scores = (self._tfidf_matrix @ q.T).toarray().ravel()

        order = np.argsort(-scores)
        out: List[dict] = []
        for idx in order:
            kb_id = self.ids[idx]
            if exclude_id and kb_id == exclude_id:
                continue
            out.append(
                {
                    "id": kb_id,
                    "score": float(scores[idx]),
                    "entry": self.entries[idx],
                }
            )
            if len(out) >= top_k:
                break
        return out


def load_llm(base_model: str, trust_remote_code: bool) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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


def build_generation_prompt(target_row: dict, retrieved: List[dict]) -> str:
    system_msg = (
        "You are a careful multiple-choice reasoner for LogicQA. "
        "Use retrieved reasoning examples as guidance, then solve the target question. "
        "Keep reasoning very short. Use at most 2 steps. "
        "Always end with exactly: Final Answer: <A/B/C/D>."
    )

    ref_lines: List[str] = []
    for i, item in enumerate(retrieved, start=1):
        e: KBEntry = item["entry"]
        ref_lines.append(
            f"[Reference {i} | id={e.id} | style={e.style} | cosine={item['score']:.4f}]\n"
            f"Question:\n{e.question}\n\n"
            f"Context (reasoning path + answer):\n{e.context}"
        )

    refs_block = "\n\n".join(ref_lines) if ref_lines else "(No references retrieved.)"

    user_msg = (
        "Retrieved examples:\n"
        f"{refs_block}\n\n"
        "Target question:\n"
        f"{to_question_block(target_row)}\n\n"
        "Required output format:\n"
        "Step 1: <short reasoning>\n"
        "Step 2: <short reasoning or omit if not needed>\n"
        "Final Answer: <A/B/C/D>"
    )

    return system_msg, user_msg


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
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text


def safe_rate(num: int, den: int) -> float:
    return (100.0 * num / den) if den else 0.0


def evaluate(
    eval_rows: List[dict],
    retriever: CosineRetriever,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    top_k: int,
    max_new_tokens: int,
    max_eval: int,
) -> Tuple[List[dict], dict]:
    rows = eval_rows[:]
    if max_eval and max_eval > 0:
        rows = rows[:max_eval]

    outputs: List[dict] = []

    for row in tqdm(rows, desc="LogicQA RAG eval"):
        ex_id = str(row.get("example_id", ""))
        gold = str(row.get("gold", "")).strip().upper()
        bucket = bucket_of(row)
        failure_category = row.get("failure_category", "other")

        retrieved = retriever.query(
            text=to_question_block(row),
            top_k=top_k,
            exclude_id=ex_id,
        )

        system_msg, user_msg = build_generation_prompt(row, retrieved)
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            system_msg=system_msg,
            user_msg=user_msg,
            max_new_tokens=max_new_tokens,
        )
        pred = extract_final_answer(generated)

        out = {
            "example_id": ex_id,
            "gold": gold,
            "pred": pred,
            "correct": bool(pred and pred == gold),
            "bucket": bucket,
            "failure_category": failure_category,
            "cot_correct": bool(row.get("cot_correct")),
            "astar_correct": bool(row.get("astar_correct")),
            "retrieved": [
                {
                    "id": item["id"],
                    "score": round(item["score"], 6),
                    "style": item["entry"].style,
                }
                for item in retrieved
            ],
            "generated_text": generated,
        }
        outputs.append(out)

    total = len(outputs)
    correct = sum(1 for x in outputs if x["correct"])

    by_bucket_total = Counter(x["bucket"] for x in outputs)
    by_bucket_correct = Counter(x["bucket"] for x in outputs if x["correct"])

    by_cat_total = Counter(x["failure_category"] for x in outputs)
    by_cat_correct = Counter(x["failure_category"] for x in outputs if x["correct"])

    metrics = {
        "n_eval": total,
        "accuracy": safe_rate(correct, total),
        "answered_count": sum(1 for x in outputs if x["pred"] is not None),
        "answered_rate": safe_rate(sum(1 for x in outputs if x["pred"] is not None), total),
        "baseline_cot_accuracy": safe_rate(sum(1 for x in outputs if x["cot_correct"]), total),
        "baseline_astar_accuracy": safe_rate(sum(1 for x in outputs if x["astar_correct"]), total),
        "bucket_accuracy": {
            k: {
                "total": by_bucket_total[k],
                "correct": by_bucket_correct[k],
                "accuracy": safe_rate(by_bucket_correct[k], by_bucket_total[k]),
            }
            for k in sorted(by_bucket_total.keys())
        },
        "category_accuracy": {
            k: {
                "total": by_cat_total[k],
                "correct": by_cat_correct[k],
                "accuracy": safe_rate(by_cat_correct[k], by_cat_total[k]),
            }
            for k in sorted(by_cat_total.keys())
        },
    }

    return outputs, metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LogicQA CoT+A* complementary RAG experiment")

    ap.add_argument(
        "--paired-jsonl",
        default="/home/dibyanayan/tanzeel/THE FINAL CALL/llama 3.1 8b/data/cot_astar_runs/logicqa_paired_all.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="/home/dibyanayan/tanzeel/THE RAG APPROACH/outputs/logicqa_rag",
    )

    ap.add_argument("--kb-split", choices=["train", "dev", "all"], default="train")
    ap.add_argument("--eval-split", choices=["train", "dev", "all"], default="all")
    ap.add_argument("--max-eval", type=int, default=0, help="0 means evaluate all selected rows")

    ap.add_argument("--retriever", choices=["auto", "sbert", "tfidf"], default="auto")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top-k", type=int, default=3)

    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument("--max-steps-per-trace", type=int, default=8)
    ap.add_argument("--max-step-chars", type=int, default=220)

    ap.add_argument("--build-kb-only", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    paired_path = Path(args.paired_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(paired_path)

    kb_entries, kb_bucket_stats = build_complementary_kb(
        rows=rows,
        kb_split=args.kb_split,
        max_steps=args.max_steps_per_trace,
        max_step_chars=args.max_step_chars,
    )

    if not kb_entries:
        raise SystemExit("No complementary KB entries were built. Check --kb-split and input data.")

    kb_jsonl_path = out_dir / "complementary_kb.jsonl"
    kb_json_path = out_dir / "complementary_kb.json"
    write_jsonl(kb_jsonl_path, [e.to_dict() for e in kb_entries])
    write_json(kb_json_path, {"entries": [e.to_dict() for e in kb_entries]})

    kb_summary = {
        "kb_size": len(kb_entries),
        "kb_split": args.kb_split,
        "bucket_breakdown": kb_bucket_stats,
        "paired_rows_total": len(rows),
    }
    write_json(out_dir / "kb_summary.json", kb_summary)

    retriever = CosineRetriever(method=args.retriever, embedding_model=args.embedding_model)
    retriever.fit(kb_entries)

    if args.build_kb_only:
        config = {
            "paired_jsonl": str(paired_path),
            "output_dir": str(out_dir),
            "kb_split": args.kb_split,
            "retriever_requested": args.retriever,
            "retriever_actual": retriever.actual_method,
            "retriever_fallback_reason": retriever.fallback_reason,
            "build_kb_only": True,
        }
        write_json(out_dir / "run_config.json", config)
        print(json.dumps({**kb_summary, **config}, indent=2))
        return

    eval_rows = split_filter(rows, args.eval_split)
    if not eval_rows:
        raise SystemExit("No rows selected for evaluation. Check --eval-split.")

    model, tokenizer = load_llm(
        base_model=args.base_model,
        trust_remote_code=bool(args.trust_remote_code),
    )

    predictions, metrics = evaluate(
        eval_rows=eval_rows,
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        max_eval=args.max_eval,
    )

    pred_path = out_dir / "eval_predictions.jsonl"
    metrics_path = out_dir / "eval_metrics.json"

    write_jsonl(pred_path, predictions)

    full_metrics = {
        **metrics,
        "paired_jsonl": str(paired_path),
        "output_dir": str(out_dir),
        "kb_split": args.kb_split,
        "eval_split": args.eval_split,
        "top_k": args.top_k,
        "retriever_requested": args.retriever,
        "retriever_actual": retriever.actual_method,
        "retriever_fallback_reason": retriever.fallback_reason,
        "embedding_model": args.embedding_model,
        "kb_size": len(kb_entries),
        "base_model": args.base_model,
        "max_new_tokens": args.max_new_tokens,
        "max_eval": args.max_eval,
    }
    write_json(metrics_path, full_metrics)

    write_json(
        out_dir / "run_config.json",
        {
            "paired_jsonl": str(paired_path),
            "output_dir": str(out_dir),
            "kb_split": args.kb_split,
            "eval_split": args.eval_split,
            "top_k": args.top_k,
            "retriever_requested": args.retriever,
            "retriever_actual": retriever.actual_method,
            "retriever_fallback_reason": retriever.fallback_reason,
            "embedding_model": args.embedding_model,
            "base_model": args.base_model,
            "max_new_tokens": args.max_new_tokens,
            "max_eval": args.max_eval,
            "max_steps_per_trace": args.max_steps_per_trace,
            "max_step_chars": args.max_step_chars,
        },
    )

    print(json.dumps(full_metrics, indent=2))
    print(f"Saved KB to: {kb_jsonl_path}")
    print(f"Saved predictions to: {pred_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
