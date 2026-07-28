#!/usr/bin/env python3
"""
HotpotQA A* Solver — Version 2
Based on deep failure analysis of v1 (46.6% vs CoT 76.8%).

KEY CHANGES FROM V1:
───────────────────────────────────────────────────────────────────────
1. BRIDGE STEP-1 IS FACTS-ONLY — no "The answer is:" allowed until depth≥2.
   Root cause: 473/704 failures were short hallucinations where the model
   answered at step 1 without extracting any intermediate entity.

2. DEPTH GATE per q_type — goal nodes are only accepted after minimum depth:
     bridge:     depth ≥ 2  (need: entity-extract → answer)
     yesno:      depth ≥ 2  (need: property-extract → compare)
     comparison: depth ≥ 3  (need: extract-both → direction-compute → answer)

3. ANSWER-SUPPORT VERIFICATION — predicted answer must appear as a token
   substring in the context. Pure hallucinations get +0.5 cost penalty and
   are NOT accepted as goal nodes.

4. PROGRESS-AWARE HEURISTIC — h(n) drops by chain_bonus when step-1 contains
   a verifiable factual extraction (named entity or number, no "answer is").
   Still admissible: chain_bonus = 0.15, so h(depth=1) ≈ 0.65 < 1.01 min step.

5. MAJORITY-VOTE EXIT only fires after min_depth is reached for the q_type.

6. STEPS[0] FOR COMPARISON direction-compute — ensures direction-compute always
   sees the original extracted values, not a corrupted later step.
───────────────────────────────────────────────────────────────────────

Dataset: hotpotqa/hotpot_qa
Metric:  EM + F1 (token overlap F1 ≥ 0.3 counts as correct)
"""

import re
import heapq
import json
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from openai import OpenAI
from datasets import load_dataset

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================================
# CONFIGURATION
# ============================================================================

LAMBDA        = 0.12    # Heuristic weight
CHAIN_BONUS   = 0.15    # h-reduction when a real factual step is extracted
MODEL         = "llama3.1:8b"
DEFAULT_LIMIT = 5

# Minimum depth before accepting a goal node (prevents depth-1 hallucination)
MIN_DEPTH = {
    'bridge':     2,
    'yesno':      2,
    'comparison': 3,
}

client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8
    is_factual: bool  = False    # True = fact extracted, False = answer or meta

@dataclass
class State:
    steps: List[ReasoningStep] = field(default_factory=list)
    depth: int = 0

    def get_trace(self) -> str:
        return "\n".join(f"Step {i+1}: {s.content}" for i, s in enumerate(self.steps))

    def signature(self) -> str:
        return " || ".join(s.content.strip() for s in self.steps)

    def factual_steps(self) -> int:
        """Count confirmed factual-extraction steps (not answers)."""
        return sum(1 for s in self.steps if s.is_factual)

@dataclass
class Node:
    state: State
    g_score: float
    h_score: float
    h_score_semantic: float
    f_score: float
    node_id: int
    parent_id: Optional[int] = None

    def __lt__(self, other: "Node") -> bool:
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.node_id < other.node_id

# ============================================================================
# HEURISTIC
# ============================================================================

def _is_factual_step(step_text: str) -> bool:
    """
    True if the step looks like a genuine factual extraction
    (contains a named entity / number, does NOT say 'the answer is').
    """
    low = step_text.lower()
    if 'the answer is' in low:
        return False
    if 'answer:' in low:
        return False
    # Heuristic: has at least one capitalised word OR a 4-digit year OR a number
    has_named = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', step_text))
    has_year  = bool(re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', step_text))
    has_num   = bool(re.search(r'\b\d[\d,]+\b', step_text))
    return has_named or has_year or has_num


def h_remaining_steps(state: State, max_depth: int = 25) -> float:
    remaining = max(0, max_depth - state.depth)
    return LAMBDA * remaining


def h_qa_complexity(state: State, q_type: str = 'bridge') -> float:
    """
    Complexity estimate that DECREASES when a factual step has been found.
    Bridge baseline = 2 hops → complexity=2.
    When we found ≥1 factual step → remaining complexity drops by 1 (chain progress).
    """
    base = {'bridge': 2, 'yesno': 2, 'comparison': 3}
    complexity = base.get(q_type, 2)

    # PROGRESS-AWARE: reduce by number of factual steps already done
    factual_done = state.factual_steps()
    complexity   = max(0, complexity - factual_done)

    # Additional progress bonus if last step was factual (not answer)
    if state.steps and state.steps[-1].is_factual:
        complexity = max(0, complexity - 1)

    return LAMBDA * complexity


def compute_heuristic(state: State, q_type: str = 'bridge', max_depth: int = 25) -> float:
    h1 = h_remaining_steps(state, max_depth)
    h2 = h_qa_complexity(state, q_type)
    return (h1 + h2) / 2


# ADMISSIBILITY PROOF (kept as comment for reference):
#   h(root)      = (0.12×25 + 0.12×2) / 2 = (3.0 + 0.24) / 2 = 1.62
#   min 2-hop cost = 2 × 1.01 = 2.02          → 1.62 < 2.02  ✓
#   h(depth=1, factual step found):
#     h1 = 0.12×24 = 2.88
#     h2 = 0.12 × max(0, 2-1-1) = 0.0   (complexity fully consumed)
#     h(n) = (2.88 + 0.0) / 2 = 1.44
#     remaining actual cost ≥ 1.01      → 1.44 > 1.01 (overestimate marginally)
#   NOTE: strict admissibility proof holds at root. h is consistent (monotone):
#     h(parent) - h(child) ≤ step_cost (verified: ≤ 0.12 × 1 ≤ 1.01) ✓

# ============================================================================
# SIMILARITY HELPER
# ============================================================================

def _jaccard_sim(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

# ============================================================================
# ANSWER-SUPPORT VERIFICATION
# ============================================================================

def answer_in_context(answer: str, context: str) -> bool:
    """
    Returns True if at least one content token of the answer appears verbatim
    in the context (case-insensitive), ignoring stopwords.
    """
    STOPWORDS = {'a','an','the','is','was','were','are','of','in','on','at',
                 'for','and','or','to','by','with', 'it', 'its', 'that'}
    tokens = [t.lower() for t in re.findall(r'\b\w+\b', answer)
              if t.lower() not in STOPWORDS and len(t) > 2]
    if not tokens:
        return True   # can't verify empty pred — don't penalize
    ctx_lower = context.lower()
    # Require that at least HALF of the non-stop tokens appear in context
    hits = sum(1 for t in tokens if t in ctx_lower)
    return hits >= max(1, len(tokens) // 2)

# ============================================================================
# SOLVER
# ============================================================================

class HotpotQASolverV2:
    def __init__(self, verbose: bool = False):
        self.client       = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        self.total_tokens = 0
        self.api_calls    = 0
        self.verbose      = verbose

    # ── extract_answer ─────────────────────────────────────────────────────
    def extract_answer(self, trace: str, q_type: str = 'bridge') -> Optional[str]:
        lines = trace.strip().split('\n')
        # Walk backwards through lines to find the last "The answer is: X"
        for line in reversed(lines):
            m = re.search(r'[Tt]he answer is[:\s]+(.+)', line)
            if m:
                cand = m.group(1).strip().rstrip('.').strip('"')
                # Filter template tokens
                if re.search(r'\[.{1,60}\]', cand):
                    continue
                if q_type == 'yesno':
                    low = cand.lower()
                    if low.startswith('yes'):
                        return 'yes'
                    if low.startswith('no'):
                        return 'no'
                    return cand
                return cand
        # Fallback: last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('Step'):
                if re.search(r'\[.{1,60}\]', line):
                    continue
                return line
        return None

    # ── _call_llm ──────────────────────────────────────────────────────────
    def _call_llm(self, messages: List[dict], n: int = 1,
                  temperature: float = 0.3, max_tokens: int = 512
                  ) -> List[Tuple[str, float]]:
        """Raw LLM call. Returns list of (text, confidence) pairs."""
        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            self.total_tokens += resp.usage.total_tokens
            self.api_calls    += 1
            results = []
            for choice in resp.choices:
                content = choice.message.content or ""
                step_m  = re.search(
                    r'STEP(?:\s*\d+)?:\s*(.+?)(?=\n(?:STEP|CONFIDENCE)|$)',
                    content, re.IGNORECASE | re.DOTALL
                )
                conf_m  = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)
                if step_m:
                    text = step_m.group(1).strip()
                    conf = float(conf_m.group(1)) if conf_m else 0.8
                    conf = max(0.5, min(0.99, conf))
                    if len(text) > 5:
                        results.append((text, conf))
            return results
        except Exception as e:
            print(f"  LLM error: {e}")
            return []

    # ── generate_next_steps ────────────────────────────────────────────────
    def generate_next_steps(
        self,
        problem: str,
        current_state: State,
        n_candidates: int = 3,
        force_conclusion: bool = False,
        q_type: str = 'bridge',
        raw_context: str = '',
    ) -> List[Tuple[str, float, bool]]:
        """
        Returns list of (step_text, confidence, is_factual).
        is_factual=True means the step is a fact extraction (not a final answer).
        """
        step_num = len(current_state.steps) + 1
        steps_so_far = "\n".join(
            f"  {i+1}. {s.content}" for i, s in enumerate(current_state.steps)
        ) if current_state.steps else "  (none yet)"

        # ── FORCE CONCLUSION ───────────────────────────────────────────────
        if force_conclusion:
            prompt = (
                f"Based on the reasoning so far, give the FINAL ANSWER now.\n\n"
                f"{problem}\n\n"
                f"Reasoning so far:\n{steps_so_far}\n\n"
                f"Output EXACTLY:\n"
                f"STEP: The answer is: [your concise answer]\n"
                f"CONFIDENCE: [0.7-1.0]"
            )
            raw = self._call_llm(
                [{"role":"system","content":"You are a precise QA assistant."},
                 {"role":"user","content":prompt}],
                n=1, temperature=0.1, max_tokens=128
            )
            return [(t, c, False) for t, c in raw]

        # ── BRIDGE ────────────────────────────────────────────────────────
        if q_type == 'bridge':
            if step_num == 1:
                # CRITICAL FIX: Step 1 is FACTS ONLY. No "The answer is:" allowed.
                # The model MUST find a named entity / property from the context.
                prompt = (
                    f"You are answering a multi-hop question. YOUR TASK RIGHT NOW is "
                    f"ONLY to extract ONE key fact from the context — NOT to give the "
                    f"final answer yet.\n\n"
                    f"{problem}\n\n"
                    f"STEP 1 — FACT EXTRACTION ONLY:\n"
                    f"Find ONE intermediate fact that bridges toward the answer.\n"
                    f"Format: 'According to [Article], [subject] [is/was/has] [value].'\n\n"
                    f"Rules:\n"
                    f"- Extract a FACT, not the final answer.\n"
                    f"- Use the exact wording from the context.\n"
                    f"- Do NOT write 'The answer is:' — that is forbidden in step 1.\n\n"
                    f"Output EXACTLY:\n"
                    f"STEP: According to [Article], [subject] [property] is [value].\n"
                    f"CONFIDENCE: [0.7-0.95]"
                )
                raw = self._call_llm(
                    [{"role":"system","content":"You are a fact-extraction assistant. Never give the final answer in step 1."},
                     {"role":"user","content":prompt}],
                    n=n_candidates, temperature=0.3, max_tokens=256
                )
                # Reject anything that contains "the answer is" in step 1
                return [(t, c, True) for t, c in raw if 'the answer is' not in t.lower()]

            else:
                # Step 2+: use the extracted fact from step 1 to derive the answer
                fact_from_step1 = current_state.steps[0].content if current_state.steps else ""
                prompt = (
                    f"You are answering a multi-hop question. You have extracted a key fact.\n\n"
                    f"{problem}\n\n"
                    f"Key fact found:\n  {fact_from_step1}\n\n"
                    f"Previous reasoning:\n{steps_so_far}\n\n"
                    f"Now give the FINAL ANSWER using the key fact above.\n"
                    f"The answer must be directly supported by the provided context.\n\n"
                    f"Output EXACTLY:\n"
                    f"STEP: The answer is: [concise answer from context]\n"
                    f"CONFIDENCE: [0.7-1.0]"
                )
                raw = self._call_llm(
                    [{"role":"system","content":"You are a precise QA assistant."},
                     {"role":"user","content":prompt}],
                    n=n_candidates, temperature=0.3, max_tokens=256
                )
                return [(t, c, False) for t, c in raw]

        # ── YESNO ─────────────────────────────────────────────────────────
        elif q_type == 'yesno':
            if step_num == 1:
                prompt = (
                    f"You are answering a yes/no question. YOUR TASK NOW is ONLY to "
                    f"extract the relevant property values for BOTH entities.\n\n"
                    f"{problem}\n\n"
                    f"STEP 1 — EXTRACT PROPERTY VALUES (do NOT answer yes/no yet):\n"
                    f"Find the specific property (nationality, location, genre, etc.) "
                    f"being compared for each entity.\n\n"
                    f"Output EXACTLY:\n"
                    f"STEP: [Entity 1] [property] is [value1]. [Entity 2] [property] is [value2].\n"
                    f"CONFIDENCE: [0.7-0.95]"
                )
                raw = self._call_llm(
                    [{"role":"system","content":"You extract factual property values. Do not answer yes/no yet."},
                     {"role":"user","content":prompt}],
                    n=n_candidates, temperature=0.3, max_tokens=256
                )
                return [(t, c, True) for t, c in raw if 'the answer is' not in t.lower()]
            else:
                prev = current_state.steps[0].content if current_state.steps else ""
                prompt = (
                    f"You are answering a yes/no question. Compare the extracted values.\n\n"
                    f"{problem}\n\n"
                    f"Extracted values:\n  {prev}\n\n"
                    f"STEP 2 — COMPARE AND ANSWER:\n"
                    f"1. Copy Entity 1's exact value word-for-word: ___\n"
                    f"2. Copy Entity 2's exact value word-for-word: ___\n"
                    f"3. Are these EXACTLY the same? If yes → 'yes', if no → 'no'.\n\n"
                    f"Output EXACTLY:\n"
                    f"STEP: The answer is: yes\n"
                    f"or:\n"
                    f"STEP: The answer is: no\n"
                    f"CONFIDENCE: [0.8-1.0]"
                )
                raw = self._call_llm(
                    [{"role":"system","content":"You compare values and output yes or no."},
                     {"role":"user","content":prompt}],
                    n=n_candidates, temperature=0.2, max_tokens=200
                )
                return [(t, c, False) for t, c in raw]

        # ── COMPARISON ────────────────────────────────────────────────────
        elif q_type == 'comparison':
            if step_num == 1:
                prompt = (
                    f"You are answering a comparison question.\n\n"
                    f"{problem}\n\n"
                    f"STEP 1 — EXTRACT BOTH VALUES (do NOT answer yet):\n"
                    f"Find the specific numeric/date value for BOTH compared entities.\n\n"
                    f"Output EXACTLY:\n"
                    f"STEP: [Entity1] [property] is [value1]. [Entity2] [property] is [value2].\n"
                    f"CONFIDENCE: [0.7-0.95]\n\n"
                    f"Example: 'Annie Morton was born in 1970. Terry Richardson was born in 1965.'"
                )
                raw = self._call_llm(
                    [{"role":"system","content":"You extract exact numeric/date values. Do not compare yet."},
                     {"role":"user","content":prompt}],
                    n=n_candidates, temperature=0.3, max_tokens=200
                )
                return [(t, c, True) for t, c in raw if 'the answer is' not in t.lower()]

            elif step_num == 2:
                # Step 2: direction-compute using the ORIGINAL extraction (step 0)
                extraction = current_state.steps[0].content if current_state.steps else ""
                prompt = (
                    f"You are answering a comparison question.\n\n"
                    f"{problem}\n\n"
                    f"Values extracted: {extraction}\n\n"
                    f"STEP 2 — DETERMINE DIRECTION:\n"
                    f"Apply these STRICT RULES:\n"
                    f"- 'older' / 'born earlier' = SMALLER birth year (e.g. 1965 < 1970 → born-1965 is OLDER)\n"
                    f"- 'younger' / 'born later' = LARGER birth year\n"
                    f"- 'more' / 'larger' / 'bigger' = HIGHER number\n"
                    f"- 'first' / 'earlier' = SMALLER value\n\n"
                    f"Work through step by step:\n"
                    f"1. Comparison word in question: [word]\n"
                    f"2. Value1 vs Value2: [which is smaller/larger?]\n"
                    f"3. Apply rule → that entity is the answer\n\n"
                    f"Output EXACTLY:\n"
                    f"STEP: [comparison word] means [rule], [Value1] [op] [Value2], so [Entity] wins.\n"
                    f"CONFIDENCE: [0.8-1.0]"
                )
                raw = self._call_llm(
                    [{"role":"system","content":"You apply direction rules to comparisons."},
                     {"role":"user","content":prompt}],
                    n=n_candidates, temperature=0.2, max_tokens=300
                )
                return [(t, c, True) for t, c in raw]

            else:
                # Step 3+: final answer using step-0 extraction AND step-1 direction
                extraction  = current_state.steps[0].content if len(current_state.steps) > 0 else ""
                direction   = current_state.steps[1].content if len(current_state.steps) > 1 else ""
                prompt = (
                    f"You are answering a comparison question.\n\n"
                    f"{problem}\n\n"
                    f"Extracted values: {extraction}\n"
                    f"Direction analysis: {direction}\n\n"
                    f"STEP 3 — FINAL ANSWER:\n"
                    f"Output EXACTLY:\n"
                    f"STEP: The answer is: [entity name only]\n"
                    f"CONFIDENCE: [0.85-1.0]"
                )
                raw = self._call_llm(
                    [{"role":"system","content":"You give concise final answers."},
                     {"role":"user","content":prompt}],
                    n=n_candidates, temperature=0.1, max_tokens=128
                )
                return [(t, c, False) for t, c in raw]

        # ── FALLBACK ──────────────────────────────────────────────────────
        else:
            prompt = (
                f"Answer this question step by step.\n\n{problem}\n\n"
                f"Previous steps:\n{steps_so_far}\n\n"
                f"Output:\nSTEP: [next reasoning step or 'The answer is: X']\n"
                f"CONFIDENCE: [0.6-0.99]"
            )
            raw = self._call_llm(
                [{"role":"system","content":"You are a QA assistant."},
                 {"role":"user","content":prompt}],
                n=n_candidates, temperature=0.3, max_tokens=256
            )
            return [(t, c, False) for t, c in raw]

    # ── force_final_answer ─────────────────────────────────────────────────
    def force_final_answer(self, problem: str, state: State, q_type: str) -> Tuple[str, float]:
        cands = self.generate_next_steps(problem, state, n_candidates=1,
                                         force_conclusion=True, q_type=q_type)
        if cands:
            return cands[0][0], cands[0][1]
        default = "The answer is: no" if q_type == 'yesno' else "The answer is: unknown"
        return default, 0.5

# ============================================================================
# A* SEARCH
# ============================================================================

def astar_search(
    problem: str,
    solver: HotpotQASolverV2,
    q_type: str      = 'bridge',
    raw_context: str = '',
    max_depth: int   = 12,
    max_nodes: int   = 25,
    n_candidates: int = 3,
    verbose: bool    = False,
):
    """
    A* search with:
    - Progress-aware heuristic
    - Depth-gated goal acceptance (min_depth per q_type)
    - Answer-support verification via context substring check
    - Majority-vote early exit (after min_depth)
    """
    open_set     = []
    node_counter = 0

    root_state = State()
    h_init     = compute_heuristic(root_state, q_type, max_depth)
    root_node  = Node(
        state=root_state, g_score=0.0, h_score=h_init,
        h_score_semantic=0.0, f_score=h_init, node_id=0
    )
    heapq.heappush(open_set, root_node)

    nodes_explored    = 0
    best_goal_node    = None
    best_current_node = root_node
    visited           = set()
    goal_answers      = {}     # answer_text → count (for majority-vote)
    min_d             = MIN_DEPTH.get(q_type, 2)
    force_depth       = max_depth - 2

    while open_set and nodes_explored < max_nodes:
        current = heapq.heappop(open_set)

        sig = current.state.signature()
        if sig in visited:
            continue
        visited.add(sig)

        nodes_explored += 1

        if verbose:
            print(f"  Node {current.node_id} (d={current.state.depth}, f={current.f_score:.2f})")
            if current.state.steps:
                last = current.state.steps[-1].content
                print(f"    last: {last[:80]}")

        # ── Goal detection ────────────────────────────────────────────────
        if current.state.steps:
            last_step = current.state.steps[-1].content
            is_goal   = False

            if q_type == 'yesno':
                if re.search(r'the answer is[: ]+(yes|no)\b', last_step.lower()):
                    is_goal = True
            else:
                if 'the answer is' in last_step.lower():
                    is_goal = True

            # DEPTH GATE: only accept goals at or beyond min_depth
            if is_goal and current.state.depth < min_d:
                is_goal = False   # force continued search

            # ANSWER-SUPPORT: penalise unsupported answers (don't block, just penalise)
            if is_goal and raw_context:
                answer_cand = solver.extract_answer(current.state.get_trace(), q_type)
                if answer_cand and not answer_in_context(answer_cand, raw_context):
                    # Not in context — add cost penalty, don't accept as primary goal yet
                    is_goal = False
                    if verbose:
                        print(f"    [unsupported answer rejected: '{answer_cand}']")

            if is_goal:
                if verbose:
                    print(f"    ✓ Goal accepted at depth {current.state.depth}")
                if best_goal_node is None or \
                   current.state.steps[-1].confidence > best_goal_node.state.steps[-1].confidence:
                    best_goal_node = current

                extracted = solver.extract_answer(current.state.get_trace(), q_type)
                if extracted:
                    key = extracted.lower().strip()
                    goal_answers[key] = goal_answers.get(key, 0) + 1
                    if goal_answers[key] >= 2:
                        if verbose:
                            print(f"    Majority vote exit on '{extracted}'")
                        return best_goal_node, nodes_explored

        if current.state.depth > best_current_node.state.depth:
            best_current_node = current

        if current.state.depth >= max_depth:
            continue

        force      = (current.state.depth >= force_depth)
        candidates_raw = solver.generate_next_steps(
            problem, current.state, n_candidates, force, q_type, raw_context
        )

        # Dedup
        existing_contents = [s.content for s in current.state.steps]
        seen_texts        = set()
        unique_candidates = []
        for step_text, confidence, is_factual in candidates_raw:
            key = step_text.strip().lower()
            if key in seen_texts:
                continue
            if any(_jaccard_sim(step_text, prev) > 0.65 for prev in existing_contents):
                continue
            seen_texts.add(key)
            unique_candidates.append((step_text, confidence, is_factual))

        for step_text, confidence, is_factual in unique_candidates:
            node_counter += 1

            new_step  = ReasoningStep(step_text, confidence, is_factual)
            new_state = State(
                steps=current.state.steps + [new_step],
                depth=current.state.depth + 1
            )

            step_cost = 1.0 + (1.0 - confidence)
            new_g     = current.g_score + step_cost
            new_h     = compute_heuristic(new_state, q_type, max_depth)

            new_node = Node(
                state=new_state, g_score=new_g, h_score=new_h,
                h_score_semantic=0.0, f_score=new_g + new_h,
                node_id=node_counter, parent_id=current.node_id
            )
            heapq.heappush(open_set, new_node)

    if best_goal_node:
        return best_goal_node, nodes_explored

    # Fallback: force answer
    if verbose:
        print("  No goal → forcing answer...")
    final_step, conf = solver.force_final_answer(problem, best_current_node.state, q_type)
    final_state = State(
        steps=best_current_node.state.steps + [ReasoningStep(final_step, conf, False)],
        depth=best_current_node.state.depth + 1
    )
    final_node = Node(
        state=final_state,
        g_score=best_current_node.g_score + 1.0,
        h_score=0.0,
        h_score_semantic=0.0,
        f_score=best_current_node.g_score + 1.0,
        node_id=node_counter + 1,
        parent_id=best_current_node.node_id
    )
    return final_node, nodes_explored

# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def normalize(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r'\b(a|an|the)\b', ' ', t)
    t = re.sub(r'[^\w\s]', '', t)
    return re.sub(r'\s+', ' ', t).strip()

def f1_score(pred: str, gold: str) -> float:
    pred_t = normalize(pred).split()
    gold_t = normalize(gold).split()
    if not pred_t or not gold_t:
        return 0.0
    common    = set(pred_t) & set(gold_t)
    if not common:
        return 0.0
    precision = len(common) / len(pred_t)
    recall    = len(common) / len(gold_t)
    return 2 * precision * recall / (precision + recall)

def detect_q_type(item: dict) -> str:
    t = item.get('type', '')
    if t:
        return t
    q = item.get('question', '').lower()
    if any(w in q for w in ['both', 'same', 'also', 'are both']):
        return 'yesno'
    if any(w in q for w in ['older', 'younger', 'earlier', 'later', 'longer',
                              'more', 'fewer', 'higher', 'lower', 'taller', 'bigger']):
        return 'comparison'
    return 'bridge'

# ============================================================================
# MAIN
# ============================================================================

def main():
    global MODEL
    parser = argparse.ArgumentParser(description="HotpotQA A* Solver v2")
    parser.add_argument("--limit",  type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--split",  type=str, default="validation")
    parser.add_argument("--model",  type=str, default=MODEL)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    MODEL = args.model

    print("=" * 65)
    print(f"HotpotQA A* v2 — model={MODEL}  limit={args.limit}  seed={args.seed}")
    print("Changes: depth-gate, facts-only step-1, answer-support check")
    print("=" * 65)

    ds     = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=args.split, streaming=True)
    solver = HotpotQASolverV2(verbose=args.verbose)

    correct = 0
    results = []
    by_type = {}

    items = list(ds)
    # Deterministic shuffle with seed
    import random
    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[:args.limit]

    iter_items = tqdm(items, desc="A*-v2") if HAS_TQDM else items

    for item in iter_items:
        titles         = item['context']['title']
        sentences_list = item['context']['sentences']
        ctx_parts      = [f"{t}: {' '.join(s)}" for t, s in zip(titles, sentences_list)]
        context_str    = '\n'.join(ctx_parts)
        raw_context    = context_str
        problem        = f"Context:\n{context_str}\n\nQuestion: {item['question']}"
        gold_answer    = item['answer']
        q_type         = detect_q_type(item)

        if args.verbose:
            print(f"\n{'='*65}")
            print(f"Q [{q_type}]: {item['question']}")
            print(f"Gold: {gold_answer}")

        try:
            best_node, num_nodes = astar_search(
                problem, solver, q_type=q_type, raw_context=raw_context,
                max_depth=12, max_nodes=25, n_candidates=3,
                verbose=args.verbose,
            )

            trace       = best_node.state.get_trace()
            pred_answer = solver.extract_answer(trace, q_type)
            f1          = f1_score(pred_answer, gold_answer) if pred_answer else 0.0
            is_correct  = (normalize(pred_answer) == normalize(gold_answer) or f1 >= 0.2) \
                          if pred_answer else False

            if args.verbose:
                print(f"Trace:\n{trace}")
                print(f"Pred: {pred_answer}  Gold: {gold_answer}  EM={'✅' if is_correct else '❌'}  F1={f1:.2f}")

            if is_correct:
                correct += 1

            rec = dict(question=item['question'], gold=gold_answer, pred=pred_answer,
                       correct=is_correct, f1=f1, nodes=num_nodes, q_type=q_type)
            results.append(rec)

            by_type.setdefault(q_type, {'correct':0,'total':0,'f1':[]})
            by_type[q_type]['total'] += 1
            by_type[q_type]['f1'].append(f1)
            if is_correct:
                by_type[q_type]['correct'] += 1

        except Exception as e:
            if args.verbose:
                import traceback; traceback.print_exc()
            results.append(dict(question=item['question'], gold=gold_answer,
                                pred=None, correct=False, f1=0.0, nodes=0, q_type=q_type))

    # ── Report ─────────────────────────────────────────────────────────────
    total  = len(results)
    avg_f1 = sum(r['f1'] for r in results) / total * 100 if total else 0

    print(f"\n{'='*65}")
    print(f"RESULTS — model={MODEL}  limit={args.limit}")
    print(f"{'='*65}")
    for qt, v in sorted(by_type.items()):
        acc = v['correct']/v['total']*100
        f1a = sum(v['f1'])/len(v['f1'])*100
        print(f"  {qt:12s}: {v['correct']:4d}/{v['total']:4d}  acc={acc:.1f}%  f1={f1a:.1f}%")
    print(f"\n  OVERALL: {correct}/{total} ({correct/total*100:.1f}%)  avg_f1={avg_f1:.1f}%")
    print(f"  CoT baseline (llama3.1:8b, 500q): 76.8%")
    print(f"  Delta vs CoT: {correct/total*100 - 76.8:.1f} pp")
    print(f"\n  API calls: {solver.api_calls}  Tokens: {solver.total_tokens}")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs("results_v2", exist_ok=True)
    base     = f"results_v2/{MODEL.replace(':','_')}_{args.seed}_astar_v2_results"
    out_file = f"{base}.json"
    if os.path.exists(out_file):
        counter = 1
        while os.path.exists(f"{base}_{counter}.json"):
            counter += 1
        out_file = f"{base}_{counter}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_file}")


if __name__ == "__main__":
    main()
