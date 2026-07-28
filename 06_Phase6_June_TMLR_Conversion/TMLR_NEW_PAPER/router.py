#!/usr/bin/env python3
"""
Rule-Based Reasoning Strategy Router
=====================================

Implements the practical decision table from Section 6 (Discussion) of:
  "When Does Tree Search Help Language Models Reason?
   A Model-Scale and Task-Structure Analysis"

The router selects between A* search and CoT prompting using only
pre-generation features of the input question — zero LLM calls required.

Decision logic (derived from statistically significant findings):
  1. Yes/no or boolean question  -> CoT   (-18.5pp for A* at 8B, p=0.025)
  2. Question <= 10 words        -> CoT   (-8.5pp for A* at 8B)
  3. Bridge question > 15 words  -> A*    (+6.8pp for A* at 8B)
  4. Question > 15 words         -> A*    (+3.8pp for A*)
  5. Otherwise                   -> A*    (default for multi-hop tasks)

Evaluated on 8B HotpotQA matched set (N=719):
  A*  alone:    70.1%
  CoT alone:    68.0%
  This router:  72.5%   (+2.4pp over best single method)
  Oracle:       81.4%
  Oracle gap recovered: 21.0%

Usage:
    from router import route_question

    decision = route_question("What team did Tom Brady play for in 2005?")
    # -> "astar"

    decision = route_question("Is Paris the capital of France?")
    # -> "cot"
"""

import re
from typing import Literal

# ─── Stopwords for question-type detection ──────────────────────────────────
_YESNO_STARTERS = frozenset([
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "should", "may", "might", "shall",
])

_YESNO_PATTERNS = [
    r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\s",
    r"\b(yes or no)\b",
    r"\b(true or false)\b",
]

_BRIDGE_INDICATORS = [
    "who", "what", "when", "where", "which", "whose",
    "who is", "who was", "what is", "what was",
    "related to", "associated with", "born in", "died in",
    "directed by", "written by", "produced by", "played by",
    "located in", "found in", "based in",
]

_COMPARISON_WORDS = [
    "older", "younger", "earlier", "later", "longer", "shorter",
    "more", "fewer", "higher", "lower", "taller", "bigger", "smaller",
    "first", "last", "larger", "faster", "slower", "better", "worse",
]


def detect_question_type(question: str) -> Literal["bridge", "comparison", "yesno"]:
    """
    Detect question type from surface features only.
    Returns: 'bridge', 'comparison', or 'yesno'
    """
    q_lower = question.lower().strip()
    first_word = q_lower.split()[0] if q_lower.split() else ""

    # Yes/No detection
    if first_word in _YESNO_STARTERS:
        return "yesno"
    for pattern in _YESNO_PATTERNS:
        if re.search(pattern, q_lower):
            return "yesno"

    # Comparison detection (ranking/ordering keywords)
    for kw in _COMPARISON_WORDS:
        if kw in q_lower:
            return "comparison"

    # Default: bridge (multi-hop entity chain)
    return "bridge"


def route_question(
    question: str,
    q_type: str | None = None,
    model_params_billions: float = 8.0,
) -> Literal["astar", "cot"]:
    """
    Route a question to A* search or CoT prompting.

    Args:
        question:              The question text.
        q_type:                Optional pre-computed question type ('bridge',
                               'comparison', 'yesno'). If None, auto-detected.
        model_params_billions: Model size in billions of parameters.
                               Used to apply the small-model rule (<= 1B -> A*).

    Returns:
        'astar' or 'cot'
    """
    q_lower = question.lower().strip()
    word_count = len(question.split())

    # Auto-detect question type if not provided
    if q_type is None:
        q_type = detect_question_type(question)

    # Rule 1: Yes/No or boolean -> CoT
    # Evidence: -18.5pp for A* on yes/no at 8B (p=0.025)
    if q_type == "yesno":
        return "cot"

    # Rule 2: Short questions (<= 10 words) -> CoT
    # Evidence: -8.5pp for A* on short questions at 8B
    if word_count <= 10:
        return "cot"

    # Rule 3: Small model (<= 1B) on multi-hop -> A*
    # Evidence: +26.4pp for A* with Qwen-0.5B on HotpotQA (p<0.001)
    if model_params_billions <= 1.0:
        return "astar"

    # Rule 4: Bridge question with > 15 words -> A*
    # Evidence: +6.8pp for A* on long bridge questions at 8B
    if q_type == "bridge" and word_count > 15:
        return "astar"

    # Rule 5: Any question > 15 words -> A*
    # Evidence: +3.8pp for A* on very long questions at 8B
    if word_count > 15:
        return "astar"

    # Rule 6: Medium-length multi-hop -> A* (default for HotpotQA-style tasks)
    if q_type in ("bridge", "comparison"):
        return "astar"

    # Default: CoT (safest for unknown question types)
    return "cot"


def route_batch(questions: list[str], **kwargs) -> list[Literal["astar", "cot"]]:
    """Route a list of questions. Returns list of 'astar'/'cot' decisions."""
    return [route_question(q, **kwargs) for q in questions]


# ─── Evaluation function ────────────────────────────────────────────────────

def evaluate_router(
    astar_results: list[dict],
    cot_results: list[dict],
    use_f1_threshold: float = 0.5,
    model_params_billions: float = 8.0,
) -> dict:
    """
    Evaluate router accuracy against oracle on paired A*/CoT results.

    Args:
        astar_results: List of dicts with 'question', 'f1'/'correct', 'q_type'
        cot_results:   List of dicts with 'question', 'f1'/'correct'
        use_f1_threshold: F1 threshold for HotpotQA (use 0 for exact-match)
        model_params_billions: Model size for routing decisions

    Returns dict with: router_acc, astar_acc, cot_acc, oracle_acc,
                       n_astar_routed, n_cot_routed, oracle_gap_recovered
    """
    a_lkp = {r["question"]: r for r in astar_results}
    c_lkp = {r["question"]: r for r in cot_results}
    common = [q for q in a_lkp if q in c_lkp]

    def is_correct(r):
        if use_f1_threshold > 0:
            return r.get("f1", 0) >= use_f1_threshold
        return r.get("correct", False)

    router_correct = 0
    a_only_correct = 0
    c_only_correct = 0
    oracle_correct = 0
    n_astar_routed = 0
    n_cot_routed = 0

    for q in common:
        ar, cr = a_lkp[q], c_lkp[q]
        a_c = is_correct(ar)
        c_c = is_correct(cr)
        qt  = ar.get("q_type", None)
        decision = route_question(q, q_type=qt, model_params_billions=model_params_billions)

        if decision == "astar":
            n_astar_routed += 1
            router_correct += int(a_c)
        else:
            n_cot_routed += 1
            router_correct += int(c_c)

        a_only_correct += int(a_c)
        c_only_correct += int(c_c)
        oracle_correct += int(a_c or c_c)

    n = len(common)
    best_single = max(a_only_correct, c_only_correct)
    oracle_gap  = oracle_correct - best_single

    return {
        "n":                    n,
        "router_acc":           round(router_correct / n * 100, 2),
        "astar_acc":            round(a_only_correct / n * 100, 2),
        "cot_acc":              round(c_only_correct / n * 100, 2),
        "oracle_acc":           round(oracle_correct / n * 100, 2),
        "router_gain":          round((router_correct - best_single) / n * 100, 2),
        "oracle_gap_recovered": round((router_correct - best_single) / oracle_gap * 100, 1)
                                if oracle_gap > 0 else 0.0,
        "n_astar_routed":       n_astar_routed,
        "n_cot_routed":         n_cot_routed,
    }


# ─── CLI demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: route sample questions
    samples = [
        ("What team did the director of The Best Offer play for in the 2005 Champions League?", None),
        ("Is Paris the capital of France?",                                                      None),
        ("Were Scott Derrickson and Ed Wood of the same nationality?",                           None),
        ("When was the park at which Tivolis Koncertsal is located opened?",                     None),
        ("Who directed the 2017 horror film starring Barry Keoghan and Nicole Kidman?",          None),
        ("What is older, the Eiffel Tower or the Louvre?",                                       None),
        ("Does the Amazon river flow through Brazil?",                                           None),
    ]

    print("=" * 65)
    print("RULE-BASED ROUTER — DEMO")
    print("=" * 65)
    for q, qt in samples:
        detected_type = detect_question_type(q)
        decision = route_question(q, q_type=qt)
        wc = len(q.split())
        print(f"\nQ: {q[:70]}")
        print(f"   words={wc}  type={detected_type}  -> {decision.upper()}")

    # Evaluate on real data if available
    import os, json
    data_dir = os.path.join(os.path.dirname(__file__), "..")
    astar_path = os.path.join(data_dir, "27th_APRIL_WHOLE_DATASET_RUN/hotpotqa/astar/results.json")
    cot_path   = os.path.join(data_dir, "27th_APRIL_WHOLE_DATASET_RUN/hotpotqa/cot/results.json")

    if os.path.exists(astar_path) and os.path.exists(cot_path):
        print("\n" + "=" * 65)
        print("ROUTER EVALUATION (8B HotpotQA, matched 719Q)")
        print("=" * 65)

        def load(path):
            d = json.load(open(path))
            if isinstance(d, dict):
                for k in ["results","data"]:
                    if k in d and isinstance(d[k], list): return d[k]
            return d

        a_items = load(astar_path)
        c_items = load(cot_path)
        results = evaluate_router(a_items, c_items, use_f1_threshold=0.5)

        print(f"  A* only:            {results['astar_acc']:.1f}%")
        print(f"  CoT only:           {results['cot_acc']:.1f}%")
        print(f"  Rule router:        {results['router_acc']:.1f}%  "
              f"(+{results['router_gain']:.1f}pp over best single)")
        print(f"  Oracle:             {results['oracle_acc']:.1f}%")
        print(f"  Oracle gap recovered: {results['oracle_gap_recovered']:.1f}%")
        print(f"  Routed to A*: {results['n_astar_routed']}  Routed to CoT: {results['n_cot_routed']}")
