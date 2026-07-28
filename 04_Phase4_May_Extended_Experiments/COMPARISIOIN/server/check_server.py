#!/usr/bin/env python3
"""
Server Health Check
====================
Verifies that Ollama is running and the required model is available
before starting the pipeline.

Usage:
    python3 server/check_server.py
"""

import sys
import os
import json
import urllib.request
import urllib.error
import time

# Allow running from project root or server/ dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

SEPARATOR = "=" * 60


def check_ollama_running() -> bool:
    """Ping the Ollama REST API root."""
    try:
        req = urllib.request.urlopen("http://localhost:11434", timeout=5)
        return True
    except Exception:
        return False


def check_model_available(model: str) -> bool:
    """Check whether the required model is pulled in Ollama."""
    try:
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(req.read())
        model_names = [m["name"] for m in data.get("models", [])]
        # Accept both "llama3.1:8b" and "llama3.1:8b-instruct-…" variants
        return any(model.split(":")[0] in n for n in model_names)
    except Exception:
        return False


def check_generate(model: str) -> tuple:
    """Send a tiny generation request to confirm the model responds."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.OLLAMA_API_KEY)
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with one word: ready"}],
            max_tokens=8,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        reply = resp.choices[0].message.content.strip()
        return True, reply, elapsed
    except Exception as e:
        return False, str(e), 0.0


def check_input_files() -> list:
    """Return list of missing input data files."""
    missing = []
    for label, path in [
        ("CoT results  ", config.COT_RESULTS_FILE),
        ("A* results   ", config.ASTAR_RESULTS_FILE),
    ]:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"  {label}: {status}  {os.path.basename(path)}")
        if not exists:
            missing.append(path)
    return missing


def check_step_files() -> list:
    """Return list of missing step scripts."""
    missing = []
    for label, path in [
        ("Step 1", config.STEP1),
        ("Step 2", config.STEP2),
        ("Step 3", config.STEP3),
        ("Step 4", config.STEP4),
    ]:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"  {label}: {status}  {os.path.basename(path)}")
        if not exists:
            missing.append(path)
    return missing


def main():
    all_ok = True

    print(SEPARATOR)
    print("  Ollama Server & Environment Check")
    print(SEPARATOR)

    # 1 — Ollama reachable
    print("\n[1] Ollama server reachable at localhost:11434 …")
    if check_ollama_running():
        print("    ✓  Ollama is UP")
    else:
        print("    ✗  Ollama is NOT running")
        print("    →  Start it with:  ollama serve")
        all_ok = False

    # 2 — Model pulled
    print(f"\n[2] Model '{config.MODEL}' available …")
    if check_model_available(config.MODEL):
        print(f"    ✓  Model found")
    else:
        print(f"    ✗  Model not found")
        print(f"    →  Pull it with:  ollama pull {config.MODEL}")
        all_ok = False

    # 3 — Live generation test
    print(f"\n[3] Live generation test …")
    ok, reply, elapsed = check_generate(config.MODEL)
    if ok:
        print(f"    ✓  Model replied: '{reply}'  ({elapsed:.1f}s)")
    else:
        print(f"    ✗  Generation failed: {reply}")
        all_ok = False

    # 4 — Input data files
    print(f"\n[4] Input data files …")
    missing_data = check_input_files()
    if missing_data:
        all_ok = False

    # 5 — Step scripts
    print(f"\n[5] Step scripts …")
    missing_scripts = check_step_files()
    if missing_scripts:
        all_ok = False

    # 6 — A* module importable
    print(f"\n[6] A* solver importable …")
    try:
        sys.path.insert(0, config.ASTAR_DIR)
        import strategyqa_astar  # noqa
        print(f"    ✓  strategyqa_astar imported (MAX_DEPTH={strategyqa_astar.MAX_DEPTH},"
              f" MAX_NODES={strategyqa_astar.MAX_NODES})")
    except Exception as e:
        print(f"    ✗  Import failed: {e}")
        all_ok = False

    # Summary
    print()
    print(SEPARATOR)
    if all_ok:
        print("  ✓  ALL CHECKS PASSED — ready to run the pipeline")
        print()
        print("  Next step:")
        print("    python3 server/run_pipeline.py")
    else:
        print("  ✗  SOME CHECKS FAILED — fix issues above before running")
    print(SEPARATOR)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
