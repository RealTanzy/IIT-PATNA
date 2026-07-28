#!/usr/bin/env python3
"""
Query-time StrategyQA RAG using FAISS retrieval over question text.

Flow:
1) Encode user question block (context + question + optional options).
2) Retrieve most similar KB question(s) from FAISS.
3) Use retrieved reasoning as context for yes/no answer generation.
"""

from __future__ import annotations

import argparse
import json
import re
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


def parse_options(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if "|||" in text:
        raw = text.split("|||")
    elif "|" in text:
        raw = text.split("|")
    elif ";" in text:
        raw = text.split(";")
    else:
        raw = text.split(",")

    return [x.strip() for x in raw if x.strip()]


def normalize_yes_no(value: str) -> Optional[str]:
    txt = (value or "").strip().lower()
    if txt in {"yes", "y", "true", "1"}:
        return "yes"
    if txt in {"no", "n", "false", "0"}:
        return "no"
    return None


def to_question_block(query: str, context: str, options: List[str]) -> str:
    parts: List[str] = []
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Question: {query}")

    if options:
        parts.append("Options:")
        for i, opt in enumerate(options):
            parts.append(f"{chr(65 + i)}. {opt}")

    return "\n".join(parts)


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None

    m = re.search(r"Final\s*Answer\s*:\s*(yes|no)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    m2 = re.findall(r"(?:the\s+)?answer\s*(?:is|:)\s*(yes|no)\b", text, flags=re.IGNORECASE)
    if m2:
        return m2[-1].lower()

    # Fallback: read the tail for standalone yes/no.
    tail = "\n".join(text.splitlines()[-4:])
    yn = re.findall(r"\b(yes|no)\b", tail, flags=re.IGNORECASE)
    if yn:
        return yn[-1].lower()

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

    k = min(max(top_k, 1), len(kb_entries))
    scores, idxs = index.search(q_emb, k)

    out: List[dict] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        entry = kb_entries[int(idx)]
        out.append(
            {
                "score": float(score),
                "entry": entry,
            }
        )
    return out


def build_prompt(target_question_block: str, retrieved: List[dict]) -> Tuple[str, str]:
    system_msg = (
        "You are a careful binary reasoner for StrategyQA. "
        "Use the retrieved reasoning context as guidance, then solve the target question. "
        "Keep reasoning short. End with exactly: Final Answer: yes or no."
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
        "Final Answer: yes or no"
    )
    return system_msg, user_msg


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    artifacts = base_dir / "artifacts"

    ap = argparse.ArgumentParser(description="StrategyQA FAISS-backed RAG query runner")
    ap.add_argument("--kb-jsonl", default=str(artifacts / "strategyqa_reasoning_kb.jsonl"))
    ap.add_argument("--index-meta", default=str(artifacts / "strategyqa_index_meta.json"))
    ap.add_argument("--faiss-index", default=str(artifacts / "strategyqa_questions.faiss"))

    ap.add_argument("--embedding-model", default="", help="Optional override. Uses index meta by default.")
    ap.add_argument("--top-k", type=int, default=1)

    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument("--query", default="")
    ap.add_argument("--context", default="")
    ap.add_argument(
        "--options",
        default="",
        help="Optional options separated by ||| (or |, ;, ,).",
    )
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")

    return ap.parse_args()


def run_once(
    args: argparse.Namespace,
    kb_entries: List[dict],
    embedding_model: str,
    query: str,
    context: str,
    options: List[str],
    model,
    tokenizer,
) -> None:
    question_block = to_question_block(query=query, context=context, options=options)
    retrieved = retrieve(
        question_block=question_block,
        kb_entries=kb_entries,
        faiss_index_path=Path(args.faiss_index),
        embedding_model=embedding_model,
        top_k=args.top_k,
    )

    print("\n=== QUERY ===")
    print(question_block)

    print("\n=== RETRIEVED REFERENCES ===")
    for i, item in enumerate(retrieved, start=1):
        e = item["entry"]
        print(f"{i}. id={e['id']} source={e['source']} score={item['score']:.4f}")

    if args.retrieval_only:
        print("\nRetrieval-only mode enabled. Skipping generation.")
        return

    system_msg, user_msg = build_prompt(target_question_block=question_block, retrieved=retrieved)
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        system_msg=system_msg,
        user_msg=user_msg,
        max_new_tokens=args.max_new_tokens,
    )
    parsed = extract_final_answer(generated)

    print("\n=== MODEL OUTPUT ===")
    print(generated)
    print("\n=== PARSED ANSWER ===")
    print(parsed if parsed else "(could not parse yes/no)")


def main() -> None:
    args = parse_args()

    meta = load_json(Path(args.index_meta))
    kb_entries = load_jsonl(Path(args.kb_jsonl))

    embedding_model = args.embedding_model.strip() or str(meta.get("embedding_model", "")).strip()
    if not embedding_model:
        raise SystemExit("Could not determine embedding model. Provide --embedding-model or valid index meta.")

    print("Loaded retrieval runtime:")
    print(
        json.dumps(
            {
                "kb_size": len(kb_entries),
                "faiss_index": args.faiss_index,
                "embedding_model": embedding_model,
                "top_k": args.top_k,
            },
            indent=2,
        )
    )

    model = tokenizer = None
    if not args.retrieval_only:
        model, tokenizer = load_generator(
            base_model=args.base_model,
            trust_remote_code=bool(args.trust_remote_code),
        )

    if args.interactive or not args.query.strip():
        print("\nInteractive mode. Submit empty question to exit.")
        while True:
            context = input("\nContext (optional): ").strip()
            query = input("Question: ").strip()
            if not query:
                print("Exiting.")
                break
            options_line = input("Options (optional, separate with |||): ").strip()
            options = parse_options(options_line)

            run_once(
                args=args,
                kb_entries=kb_entries,
                embedding_model=embedding_model,
                query=query,
                context=context,
                options=options,
                model=model,
                tokenizer=tokenizer,
            )
        return

    run_once(
        args=args,
        kb_entries=kb_entries,
        embedding_model=embedding_model,
        query=args.query.strip(),
        context=args.context.strip(),
        options=parse_options(args.options),
        model=model,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
