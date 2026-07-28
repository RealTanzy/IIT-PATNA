#!/usr/bin/env python3
"""
Single-query tester for the LogicQA complementary RAG pipeline.

Usage examples:

1) One-shot query:
python3 "THE RAG APPROACH/logicqa_rag_query_test.py" \
  --query "Which option best weakens the argument?" \
  --context "A study claims X because Y." \
  --options "Option one|||Option two|||Option three|||Option four"

2) Interactive mode:
python3 "THE RAG APPROACH/logicqa_rag_query_test.py" --interactive
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import List, Tuple


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


def load_core_module(script_path: Path):
    core_path = script_path.with_name("logicqa_rag_experiment.py")
    spec = importlib.util.spec_from_file_location("logicqa_rag_experiment", core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load core module from {core_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_kb_entries_from_jsonl(core, kb_jsonl: Path):
    rows = core.load_jsonl(kb_jsonl)
    entries = []
    for r in rows:
        entries.append(
            core.KBEntry(
                id=str(r.get("id", "")),
                question=str(r.get("question", "")),
                context=str(r.get("context", "")),
                style=str(r.get("style", "")),
                source_bucket=str(r.get("source_bucket", "")),
                answer=str(r.get("answer", "")),
            )
        )
    return entries


def prepare_runtime(core, args):
    if args.kb_jsonl:
        kb_path = Path(args.kb_jsonl)
        if not kb_path.exists():
            raise FileNotFoundError(f"KB JSONL not found: {kb_path}")
        kb_entries = load_kb_entries_from_jsonl(core, kb_path)
    else:
        paired_path = Path(args.paired_jsonl)
        rows = core.load_jsonl(paired_path)
        kb_entries, _ = core.build_complementary_kb(
            rows=rows,
            kb_split=args.kb_split,
            max_steps=args.max_steps_per_trace,
            max_step_chars=args.max_step_chars,
        )

    if not kb_entries:
        raise RuntimeError("No KB entries available. Check kb source and split.")

    retriever = core.CosineRetriever(method=args.retriever, embedding_model=args.embedding_model)
    retriever.fit(kb_entries)

    model, tokenizer = core.load_llm(
        base_model=args.base_model,
        trust_remote_code=bool(args.trust_remote_code),
    )

    return retriever, model, tokenizer, kb_entries


def run_single_query(
    core,
    retriever,
    model,
    tokenizer,
    query: str,
    context: str,
    options: List[str],
    top_k: int,
    max_new_tokens: int,
    show_references: bool,
) -> Tuple[str, str]:
    row = {
        "example_id": "adhoc_query",
        "context": context,
        "query": query,
        "options": options,
    }

    question_block = core.to_question_block(row)
    retrieved = retriever.query(text=question_block, top_k=top_k, exclude_id=None)

    system_msg, user_msg = core.build_generation_prompt(row, retrieved)
    generated = core.generate_text(
        model=model,
        tokenizer=tokenizer,
        system_msg=system_msg,
        user_msg=user_msg,
        max_new_tokens=max_new_tokens,
    )
    pred = core.extract_final_answer(generated)

    print("\n=== QUERY ===")
    print(question_block)

    if show_references:
        print("\n=== RETRIEVED REFERENCES ===")
        for i, item in enumerate(retrieved, start=1):
            entry = item["entry"]
            print(
                f"{i}. id={item['id']} score={item['score']:.4f} "
                f"style={entry.style} bucket={entry.source_bucket}"
            )

    print("\n=== MODEL OUTPUT ===")
    print(generated)

    print("\n=== PARSED ANSWER ===")
    print(pred if pred else "(could not parse A/B/C/D)")

    return generated, pred


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ad-hoc query tester for LogicQA complementary RAG")

    ap.add_argument(
        "--paired-jsonl",
        default="/home/dibyanayan/tanzeel/THE FINAL CALL/llama 3.1 8b/data/cot_astar_runs/logicqa_paired_all.jsonl",
    )
    ap.add_argument("--kb-jsonl", default="", help="Optional prebuilt KB JSONL path")
    ap.add_argument("--kb-split", choices=["train", "dev", "all"], default="train")

    ap.add_argument("--retriever", choices=["auto", "sbert", "tfidf"], default="auto")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top-k", type=int, default=3)

    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument("--max-steps-per-trace", type=int, default=8)
    ap.add_argument("--max-step-chars", type=int, default=220)

    ap.add_argument("--query", default="")
    ap.add_argument("--context", default="")
    ap.add_argument(
        "--options",
        default="",
        help="Options separated by ||| (or |, ;, ,). Example: optA|||optB|||optC|||optD",
    )
    ap.add_argument("--interactive", action="store_true")

    ap.add_argument("--show-references", action="store_true", default=True)
    ap.add_argument("--hide-references", action="store_true", help="Do not print retrieved references")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    show_references = bool(args.show_references) and not bool(args.hide_references)

    this_script = Path(__file__).resolve()
    core = load_core_module(this_script)

    retriever, model, tokenizer, kb_entries = prepare_runtime(core, args)

    print("Loaded runtime:")
    print(json.dumps({
        "kb_size": len(kb_entries),
        "retriever_requested": args.retriever,
        "retriever_actual": retriever.actual_method,
        "retriever_fallback_reason": retriever.fallback_reason,
        "base_model": args.base_model,
    }, indent=2))

    if args.interactive or not args.query.strip():
        print("\nInteractive mode. Submit empty question to exit.")
        while True:
            context = input("\nContext (optional): ").strip()
            query = input("Question: ").strip()
            if not query:
                print("Exiting.")
                break
            options_line = input("Options A-D (optional, separate with |||): ").strip()
            options = parse_options(options_line)

            run_single_query(
                core=core,
                retriever=retriever,
                model=model,
                tokenizer=tokenizer,
                query=query,
                context=context,
                options=options,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                show_references=show_references,
            )
        return

    options = parse_options(args.options)
    run_single_query(
        core=core,
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        query=args.query.strip(),
        context=args.context.strip(),
        options=options,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        show_references=show_references,
    )


if __name__ == "__main__":
    main()
