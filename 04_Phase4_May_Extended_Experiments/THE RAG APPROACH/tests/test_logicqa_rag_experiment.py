import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "logicqa_rag_experiment.py"
PAIRED_JSONL = Path(
    "/home/dibyanayan/tanzeel/THE FINAL CALL/llama 3.1 8b/data/cot_astar_runs/logicqa_paired_all.jsonl"
)


spec = importlib.util.spec_from_file_location("logicqa_rag_experiment", MODULE_PATH)
rag = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rag)


def test_extract_final_answer_variants():
    assert rag.extract_final_answer("Final Answer: C") == "C"
    assert rag.extract_final_answer("The answer is b") == "B"
    assert rag.extract_final_answer("Option D is correct") == "D"
    assert rag.extract_final_answer("Step 1: short\nA") == "A"


def test_linearize_reasoning_dedup_and_final_answer():
    trace = """
    Step 1: Alpha reasoning segment with enough detail to keep.
    Step 2: Alpha reasoning segment with enough detail to keep.
    Step 3: Beta reasoning segment that should remain.
    """
    out = rag.linearize_reasoning(trace=trace, final_answer="B", max_steps=5, max_step_chars=120)
    assert "Final Answer: B" in out
    assert out.count("Step ") <= 2


@pytest.mark.skipif(not PAIRED_JSONL.exists(), reason="Paired LogicQA JSONL not found in workspace")
def test_build_complementary_kb_train_split_counts():
    rows = rag.load_jsonl(PAIRED_JSONL)
    entries, stats = rag.build_complementary_kb(
        rows=rows,
        kb_split="train",
        max_steps=8,
        max_step_chars=220,
    )

    # Expected from current paired file composition.
    assert len(entries) == 25
    assert stats["astar_only"] == 10
    assert stats["cot_only"] == 15

    # JSON structure sanity.
    sample = entries[0].to_dict()
    assert set(sample.keys()) == {"id", "question", "context", "style", "source_bucket", "answer"}


def test_retriever_tfidf_excludes_self():
    entries = [
        rag.KBEntry(
            id="q1",
            question="Context: X\nQuestion: Which option says cat?",
            context="Step 1: cat\nFinal Answer: A",
            style="cot_preserved",
            source_bucket="cot_only",
            answer="A",
        ),
        rag.KBEntry(
            id="q2",
            question="Context: Y\nQuestion: Which option says dog?",
            context="Step 1: dog\nFinal Answer: B",
            style="astar_linearized",
            source_bucket="astar_only",
            answer="B",
        ),
        rag.KBEntry(
            id="q3",
            question="Context: Z\nQuestion: Which option says bird?",
            context="Step 1: bird\nFinal Answer: C",
            style="cot_preserved",
            source_bucket="cot_only",
            answer="C",
        ),
    ]

    retriever = rag.CosineRetriever(method="tfidf", embedding_model="unused")
    retriever.fit(entries)
    hits = retriever.query("Question: Which option says cat?", top_k=2, exclude_id="q1")

    assert len(hits) == 2
    assert all(h["id"] != "q1" for h in hits)


@pytest.mark.skipif(not PAIRED_JSONL.exists(), reason="Paired LogicQA JSONL not found in workspace")
def test_cli_build_kb_only_smoke(tmp_path):
    out_dir = tmp_path / "rag_smoke"
    cmd = [
        sys.executable,
        str(MODULE_PATH),
        "--build-kb-only",
        "--retriever",
        "tfidf",
        "--kb-split",
        "train",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "complementary_kb.jsonl").exists()
    assert (out_dir / "complementary_kb.json").exists()
    assert (out_dir / "kb_summary.json").exists()
    assert (out_dir / "run_config.json").exists()

    summary = json.loads((out_dir / "kb_summary.json").read_text(encoding="utf-8"))
    assert summary["kb_size"] > 0
