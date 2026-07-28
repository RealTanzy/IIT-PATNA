#!/usr/bin/env python3
"""
Full Pipeline Runner
=====================
Runs the complete CoT vs A* comparison experiment end-to-end.

What it does:
  Step 1  Find disagreements between original CoT and A* runs
          → 49 questions CoT failed / A* won  (re-run with CoT)
          → 45 questions A* failed / CoT won  (re-run with A*)

  Step 2  Re-run CoT on the 49 questions it originally failed
          Each question gets 2 independent CoT attempts (temp 0.2 and 0.5)

  Step 3  Re-run A* on the 45 questions it originally failed
          Full A* search (branch_k=2, max_nodes=15) per question

  Step 4  Final comparison report

Usage:
    python3 server/run_pipeline.py              # run all steps
    python3 server/run_pipeline.py --from 2     # resume from step 2
    python3 server/run_pipeline.py --only 4     # run only step 4 (report)
    python3 server/run_pipeline.py --check      # health check only
"""

import sys
import os
import argparse
import subprocess
import time
import json

# Ensure config is importable from project root or server/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

SEPARATOR = "=" * 65
PYTHON    = config.VENV_PYTHON


def _banner(step: int, title: str):
    print()
    print(SEPARATOR)
    print(f"  STEP {step} — {title}")
    print(SEPARATOR)


def _run_script(path: str, step_name: str) -> bool:
    """Run a Python script and stream its output. Returns True on success."""
    print(f"\n  Running: {os.path.basename(path)}")
    print(f"  Path   : {path}")
    print()
    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, path],
            check=False,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n  ✓ {step_name} completed in {elapsed:.0f}s")
            return True
        else:
            print(f"\n  ✗ {step_name} exited with code {result.returncode}")
            return False
    except FileNotFoundError:
        print(f"\n  ✗ Python not found at: {PYTHON}")
        print(f"    Edit server/config.py → VENV_PYTHON")
        return False
    except KeyboardInterrupt:
        print(f"\n\n  ⚠ Interrupted. Partial results saved incrementally.")
        sys.exit(1)


def _check_output_exists(path: str, label: str) -> bool:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        n = len(data)
        print(f"  ✓ {label} already exists ({n} items) — skipping")
        return True
    return False


def run_health_check() -> bool:
    """Run check_server.py and return whether it passed."""
    check_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_server.py")
    result = subprocess.run([PYTHON, check_path], check=False)
    return result.returncode == 0


# ── Step functions ────────────────────────────────────────────────────────────

def step1():
    _banner(1, "Find Disagreements")
    print("  Identifies:")
    print("    A) CoT failed + A* won  →  cot_fail_astar_win.json")
    print("    B) A* failed + CoT won  →  astar_fail_cot_win.json")
    print("    C) Both failed          →  both_failed.json")
    print("    D) Both correct         →  both_correct.json")

    # Check if already done
    if all(os.path.exists(p) for p in [
        config.COT_FAIL_ASTAR_WIN, config.ASTAR_FAIL_COT_WIN,
        config.BOTH_FAILED, config.BOTH_CORRECT
    ]):
        with open(config.COT_FAIL_ASTAR_WIN) as f: n1 = len(json.load(f))
        with open(config.ASTAR_FAIL_COT_WIN)  as f: n2 = len(json.load(f))
        print(f"\n  ✓ Step 1 outputs already exist")
        print(f"    CoT-fail/A*-win  : {n1} questions")
        print(f"    A*-fail/CoT-win  : {n2} questions")
        return True

    return _run_script(config.STEP1, "Step 1")


def step2():
    _banner(2, "Re-run CoT on its failures (49 questions)")
    print("  Each question: 2 independent CoT calls (temp 0.2 and 0.5)")
    print("  Expected time: ~5–10 min")
    print(f"  Output: cot_rerun_on_cot_failures.json")

    if _check_output_exists(config.COT_RERUN_OUT, "cot_rerun_on_cot_failures.json"):
        return True

    return _run_script(config.STEP2, "Step 2")


def step3():
    _banner(3, "Re-run A* on its failures (45 questions)")
    print("  Each question: full A* search (branch_k=2, max_nodes=15)")
    print("  Expected time: ~20–35 min")
    print(f"  Output: astar_rerun_on_astar_failures.json")
    print(f"  Note  : saves incrementally — safe to interrupt and resume")

    if _check_output_exists(config.ASTAR_RERUN_OUT, "astar_rerun_on_astar_failures.json"):
        return True

    return _run_script(config.STEP3, "Step 3")


def step4():
    _banner(4, "Final Comparison Report")
    print("  Compares baseline vs re-run accuracy for both methods")
    return _run_script(config.STEP4, "Step 4")


# ── Main ──────────────────────────────────────────────────────────────────────

STEPS = {1: step1, 2: step2, 3: step3, 4: step4}

def main():
    parser = argparse.ArgumentParser(description="CoT vs A* Pipeline Runner")
    parser.add_argument("--from",   dest="from_step", type=int, default=1,
                        help="Start from this step (1-4). Default: 1")
    parser.add_argument("--only",   dest="only_step", type=int, default=None,
                        help="Run only this step (1-4)")
    parser.add_argument("--check",  action="store_true",
                        help="Run health check only")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip the health check at the start")
    args = parser.parse_args()

    print()
    print(SEPARATOR)
    print("  CoT vs A* Comparison — Full Pipeline")
    print(f"  Model : {config.MODEL}")
    print(f"  Server: {config.OLLAMA_BASE_URL}")
    print(SEPARATOR)

    # Health check
    if args.check:
        sys.exit(0 if run_health_check() else 1)

    if not args.no_check:
        print("\n  Running health check first …\n")
        if not run_health_check():
            print("\n  ✗ Health check failed. Fix issues and re-run.")
            print("    Or skip with:  python3 server/run_pipeline.py --no-check")
            sys.exit(1)

    # Determine which steps to run
    if args.only_step:
        steps_to_run = [args.only_step]
    else:
        steps_to_run = list(range(args.from_step, 5))

    print(f"\n  Steps to run: {steps_to_run}")

    pipeline_start = time.time()
    failed_at = None

    for s in steps_to_run:
        if s not in STEPS:
            print(f"  ✗ Unknown step: {s}")
            sys.exit(1)
        ok = STEPS[s]()
        if not ok:
            failed_at = s
            break

    elapsed = time.time() - pipeline_start
    print()
    print(SEPARATOR)
    if failed_at:
        print(f"  ✗ Pipeline stopped at step {failed_at}  (total: {elapsed:.0f}s)")
        print(f"    Resume with:  python3 server/run_pipeline.py --from {failed_at}")
    else:
        print(f"  ✓ Pipeline complete  (total: {elapsed:.0f}s)")
        print()
        print("  Output files:")
        for label, path in [
            ("Disagreements (CoT fail / A* win)", config.COT_FAIL_ASTAR_WIN),
            ("Disagreements (A* fail / CoT win)", config.ASTAR_FAIL_COT_WIN),
            ("CoT re-run results               ", config.COT_RERUN_OUT),
            ("A* re-run results                ", config.ASTAR_RERUN_OUT),
        ]:
            exists = "✓" if os.path.exists(path) else "✗ missing"
            print(f"    {exists}  {label:36s}  {os.path.basename(path)}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
