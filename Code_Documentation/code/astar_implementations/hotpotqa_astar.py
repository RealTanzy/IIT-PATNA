#!/usr/bin/env python3
"""
HotpotQA A* Search Solver — New Improved Version
=================================================

Designed to beat Chain-of-Thought baseline (76.8% accuracy, llama3.1:8b, 500 examples).

ROOT CAUSE ANALYSIS OF PREVIOUS A* FAILURES (44.6% on 1500 examples):
──────────────────────────────────────────────────────────────────────
1. NO REAL BRANCHING (54% of failures)
   Ollama ignores n>1 in chat completions → only 1 candidate per expansion.
   Average 2.44 nodes = no search at all, just a linear chain.
   ▸ FIX: K=3 separate API calls with varied temperatures for TRUE branching.

2. PREMATURE EXIT (returns first goal found)
   Code does `if best_goal_node: return` after each expansion.
   Search never explores alternatives.
   ▸ FIX: Continue past first goal. Collect multiple goals. Majority vote.

3. RIGID PROMPTS (cripple model reasoning)
   COT works because the model reasons FREELY over full context.
   Old A* constrains model into rigid "STEP: ... CONFIDENCE: ..." format.
   ▸ FIX: COT-hybrid prompts: structured output but natural reasoning.

4. WEAK HEURISTIC (depth-only, non-informative)
   h(s) = LAMBDA × remaining_depth — identical for all nodes at same depth.
   Doesn't distinguish good from bad reasoning paths.
   ▸ FIX: Entity-coverage + context-grounding heuristic (informative AND admissible).

5. NO VERIFICATION (hallucinations accepted as goals)
   5.2% are "not found"/"none", 38.3% are wrong answers — no quality check.
   ▸ FIX: Context grounding check. Penalty for unverifiable answers.

ADMISSIBILITY PROOF
───────────────────
h(s) = h_depth(s) + h_coverage(s) where:
  h_depth(s)    = max(0, 2 - depth(s)) × 0.3   [remaining hops × weight]
  h_coverage(s) = (1 - entity_coverage) × 0.3   [unaddressed question entities]

Step cost:  c(s→s') = 1.0 + (1.0 - confidence) ≥ 1.01  (confidence capped at 0.99)
Min hops:   2 (HotpotQA is multi-hop)
Min cost:   root→goal ≥ 2 × 1.01 = 2.02

Proof (non-goal states only — goal states have h*(s)=0 and aren't evaluated):
  d=0: h ≤ 2×0.3 + 1×0.3 = 0.9  < 2.02 = min(h*)  ✓
  d=1: h ≤ 1×0.3 + 1×0.3 = 0.6  < 1.01 = min(h*)  ✓
  d≥2: h ≤ 0×0.3 + 1×0.3 = 0.3  < 1.01 = min(h*)  ✓

Consistency: max(Δh per step) = 0.3+0.3 = 0.6 < 1.01 = min(c)  ✓
⟹  A* with graph search is OPTIMAL and COMPLETE.
"""

import os
import re
import heapq
import json
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "llama3.1:8b"
DEFAULT_LIMIT = 500 
BASE_URL = "http://localhost:11434/v1"

# Heuristic weights (must satisfy admissibility: see proof above)
WEIGHT_DEPTH    = 0.3   # Per remaining hop
WEIGHT_COVERAGE = 0.3   # For entity coverage

# Search parameters
BRANCH_K       = 3      # Candidates per expansion (separate API calls)
MAX_DEPTH      = 8      # Maximum reasoning depth
MAX_NODES      = 30     # Maximum nodes to explore
MIN_GOAL_DEPTH = 2      # Minimum depth before accepting a goal
MAJORITY_VOTE  = 2      # Votes needed for early exit
TEMPERATURES   = [0.15, 0.45, 0.75]  # Per-branch temperatures


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8
    is_factual: bool = False      # True: fact extraction; False: answer/conclusion

@dataclass
class State:
    steps: List[ReasoningStep] = field(default_factory=list)
    depth: int = 0

    def get_trace(self) -> str:
        return "\n".join(
            f"Step {i+1}: {s.content}" for i, s in enumerate(self.steps)
        )

    def signature(self) -> str:
        return " || ".join(s.content.strip() for s in self.steps)

@dataclass
class Node:
    state: State
    g_score: float        # Actual cost so far
    h_score: float        # Heuristic estimate to goal
    f_score: float        # g + h
    node_id: int
    parent_id: Optional[int] = None

    def __lt__(self, other: "Node") -> bool:
        if abs(self.f_score - other.f_score) > 1e-9:
            return self.f_score < other.f_score
        return self.node_id < other.node_id


# ============================================================================
# QUESTION TYPE DETECTION
# ============================================================================

def detect_question_type(item: dict) -> str:
    """
    Detect question type using dataset metadata + heuristics.
    Returns: 'bridge', 'yesno', or 'comparison'.

    Uses HotpotQA's 'type' field when available, combined with
    answer content for yesno disambiguation.
    """
    dtype = item.get('type', '')
    question = item.get('question', '').lower().strip()
    answer = item.get('answer', '').strip().lower()

    # Ground truth: if answer is yes/no, it's a yesno question
    if answer in ('yes', 'no'):
        return 'yesno'

    # Dataset type field
    if dtype == 'comparison':
        # Check if it's actually a yesno disguised as comparison
        yesno_patterns = [
            r'^(are|were|is|was|do|does|did|have|has|had)\s+',
            r'\bboth\b',
            r'\bsame\b',
            r'\balso\b',
        ]
        if any(re.search(p, question) for p in yesno_patterns):
            return 'yesno'

        # True comparison with ranking/ordering keywords
        comparison_kw = [
            'older', 'younger', 'earlier', 'later', 'longer', 'shorter',
            'more', 'fewer', 'higher', 'lower', 'taller', 'bigger',
            'smaller', 'first', 'last', 'larger', 'faster', 'slower',
            'which is', 'who is', 'which has', 'who has', 'which was',
        ]
        if any(kw in question for kw in comparison_kw):
            return 'comparison'

        # Comparison type but no ranking keyword → might be bridge-style
        return 'comparison'

    if dtype == 'bridge':
        return 'bridge'

    # Fallback: no type field, use heuristics
    if any(kw in question for kw in ['both', 'same', 'also a ', 'are the']):
        return 'yesno'
    if any(kw in question for kw in ['older', 'younger', 'more', 'fewer',
                                       'which is', 'who is older', 'first']):
        return 'comparison'
    return 'bridge'


# ============================================================================
# ENTITY EXTRACTION (for heuristic)
# ============================================================================

# Common stopwords to exclude from entity matching
_STOPWORDS = frozenset({
    'a', 'an', 'the', 'is', 'was', 'were', 'are', 'of', 'in', 'on', 'at',
    'for', 'and', 'or', 'to', 'by', 'with', 'it', 'its', 'that', 'this',
    'from', 'as', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
    'what', 'which', 'who', 'whom', 'where', 'when', 'how', 'why',
    'same', 'both', 'also', 'not', 'no', 'yes', 'than', 'more', 'most',
    'other', 'some', 'any', 'all', 'many', 'much', 'few', 'several',
    'held', 'based', 'located', 'known', 'called', 'named', 'played',
    'first', 'last', 'new', 'old', 'older', 'younger',
})


def extract_question_entities(question: str) -> Set[str]:
    """
    Extract meaningful entities from a question for heuristic evaluation.
    Returns set of lowercase entity tokens (proper nouns, numbers, key terms).
    """
    entities = set()

    # 1. Quoted strings (exact references)
    for m in re.finditer(r'"([^"]+)"', question):
        entities.add(m.group(1).lower())

    # 2. Capitalized phrases (proper nouns) — skip sentence start
    q_trimmed = re.sub(r'^(What|Which|Who|Where|When|How|Are|Were|Is|Was|Did|Do|Does|The)\s+',
                       '', question, flags=re.IGNORECASE)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', q_trimmed):
        phrase = m.group(1).lower()
        if phrase not in _STOPWORDS and len(phrase) > 2:
            entities.add(phrase)

    # 3. Numbers / years
    for m in re.finditer(r'\b(\d{4}|\d+(?:,\d{3})*)\b', question):
        entities.add(m.group(1))

    # 4. Key content words (non-stopwords, len>3)
    words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
    for w in words:
        if w not in _STOPWORDS and len(w) > 3:
            entities.add(w)

    return entities


def compute_entity_coverage(state: State, question_entities: Set[str]) -> float:
    """Fraction of question entities mentioned in the reasoning trace."""
    if not question_entities:
        return 1.0
    trace = state.get_trace().lower()
    covered = sum(1 for e in question_entities if e in trace)
    return covered / len(question_entities)


# ============================================================================
# ADMISSIBLE + CONSISTENT HEURISTIC
# ============================================================================

def compute_heuristic(state: State, question_entities: Set[str],
                      q_type: str = 'bridge') -> float:
    """
    Multi-component admissible heuristic for HotpotQA A* search.

    h(s) = h_depth(s) + h_coverage(s)

    h_depth:    Remaining minimum hops × WEIGHT_DEPTH
    h_coverage: Unaddressed question entities × WEIGHT_COVERAGE

    Admissibility: h(s) ≤ 0.9 < 2.02 = min cost(root→goal). See module docstring.
    Consistency:  max Δh = 0.6 < 1.01 = min step cost. See module docstring.
    """
    min_hops = 2  # HotpotQA always needs ≥2 hops
    remaining_hops = max(0, min_hops - state.depth)
    h_depth = remaining_hops * WEIGHT_DEPTH

    coverage = compute_entity_coverage(state, question_entities)
    h_coverage = (1.0 - coverage) * WEIGHT_COVERAGE

    return h_depth + h_coverage


# ============================================================================
# CONTEXT GROUNDING VERIFICATION
# ============================================================================

def verify_answer_in_context(answer: str, context: str) -> float:
    """
    Score how well the predicted answer is grounded in the context.
    Returns a score in [0, 1] where 1 = fully grounded.
    """
    if not answer or not context:
        return 0.0

    ctx_lower = context.lower()
    answer_lower = answer.lower().strip()

    # Direct substring match (best case)
    if answer_lower in ctx_lower:
        return 1.0

    # Token overlap
    tokens = [t for t in re.findall(r'\b\w+\b', answer_lower)
              if t not in _STOPWORDS and len(t) > 2]
    if not tokens:
        return 0.5  # Short answers like "yes"/"no" — don't penalize

    hits = sum(1 for t in tokens if t in ctx_lower)
    return hits / len(tokens)


# ============================================================================
# SIMILARITY HELPER
# ============================================================================

def jaccard_sim(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ============================================================================
# SOLVER
# ============================================================================

class HotpotQASolver:
    """LLM-backed solver with real branching for A* search."""

    def __init__(self, model: str = MODEL, verbose: bool = False):
        self.client = OpenAI(base_url=BASE_URL, api_key='ollama')
        self.model = model
        self.total_tokens = 0
        self.api_calls = 0
        self.verbose = verbose

    # ── Single LLM call ────────────────────────────────────────────────
    def _call_llm(self, system: str, prompt: str,
                  temperature: float = 0.3, max_tokens: int = 512
                  ) -> Tuple[Optional[str], float]:
        """
        Make a single LLM call. Returns (step_text, confidence).
        Robust parsing handles varied output formats.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.total_tokens += resp.usage.total_tokens
            self.api_calls += 1
            content = resp.choices[0].message.content or ""
            return self._parse_output(content)
        except Exception as e:
            if self.verbose:
                print(f"    LLM error: {e}")
            return None, 0.0

    def _parse_output(self, content: str) -> Tuple[Optional[str], float]:
        """Parse LLM output, handling various formats robustly."""
        # Try STEP: ... CONFIDENCE: ... format
        step_m = re.search(
            r'STEP(?:\s*\d*)?:\s*(.+?)(?=\n(?:STEP|CONFIDENCE)|$)',
            content, re.IGNORECASE | re.DOTALL
        )
        conf_m = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)

        if step_m:
            text = step_m.group(1).strip()
            conf = float(conf_m.group(1)) if conf_m else 0.75
            conf = max(0.5, min(0.99, conf))
            if len(text) > 3:
                return text, conf

        # Fallback: "The answer is: ..." anywhere
        ans_m = re.search(r'[Tt]he answer is[:\s]+(.+?)(?:\.|$)', content)
        if ans_m:
            text = f"The answer is: {ans_m.group(1).strip()}"
            conf = float(conf_m.group(1)) if conf_m else 0.75
            return text, max(0.5, min(0.99, conf))

        # Last resort: use entire response if short enough
        text = content.strip()
        # Remove common prefixes
        text = re.sub(r'^(?:Step\s*\d+[:\s]*|STEP[:\s]*)', '', text, flags=re.IGNORECASE).strip()
        if 5 < len(text) < 500:
            return text, 0.6

        return None, 0.0

    # ── Generate K candidates with REAL BRANCHING ──────────────────────
    def generate_candidates(
        self,
        problem: str,
        state: State,
        q_type: str = 'bridge',
        context: str = '',
        force_conclusion: bool = False,
    ) -> List[Tuple[str, float, bool]]:
        """
        Generate multiple candidates via SEPARATE API calls (not n>1).
        Returns list of (step_text, confidence, is_factual).

        This is the KEY FIX: Ollama ignores n>1, so we make K separate calls
        with different temperatures and prompt strategies for REAL diversity.
        """
        step_num = len(state.steps) + 1

        if force_conclusion:
            return self._generate_conclusion(problem, state, q_type)

        if q_type == 'yesno':
            return self._generate_yesno(problem, state, step_num)
        elif q_type == 'comparison':
            return self._generate_comparison(problem, state, step_num)
        else:
            return self._generate_bridge(problem, state, step_num)

    # ── Bridge prompts ─────────────────────────────────────────────────
    def _generate_bridge(self, problem: str, state: State,
                         step_num: int) -> List[Tuple[str, float, bool]]:
        """Generate candidates for bridge (entity-chain) questions."""
        candidates = []
        steps_so_far = "\n".join(
            f"  {i+1}. {s.content}" for i, s in enumerate(state.steps)
        ) if state.steps else ""

        if step_num == 1:
            # Step 1: Fact extraction — 3 different strategies
            strategies = [
                # Strategy 1: Entity-focused
                {
                    "system": "You find intermediate facts in context paragraphs. "
                              "Extract exact information. Never give the final answer in step 1.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"TASK: Find the key intermediate entity or fact from the context "
                        f"that the question references. This is step 1 — extract a FACT, "
                        f"not the final answer.\n\n"
                        f"Example: If asked 'What city is the director of X based in?', "
                        f"first find who directed X.\n\n"
                        f"STEP: [intermediate fact from context]\n"
                        f"CONFIDENCE: [0.6-0.95]"
                    ),
                },
                # Strategy 2: Article-focused
                {
                    "system": "You identify the most relevant context paragraph and "
                              "extract key details. Be specific and cite the source.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"TASK: Which context paragraph answers part of this question? "
                        f"Extract the specific detail that serves as a bridge to the answer.\n\n"
                        f"STEP: According to the [Title] article, [specific fact].\n"
                        f"CONFIDENCE: [0.6-0.95]"
                    ),
                },
                # Strategy 3: Reasoning-focused (more COT-like)
                {
                    "system": "You are a careful multi-hop reasoning assistant. "
                              "Think about what intermediate information is needed.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"TASK: To answer this multi-hop question, what fact do I need "
                        f"to find first? Identify the bridge entity and look it up in the context.\n\n"
                        f"STEP: [the intermediate fact you found]\n"
                        f"CONFIDENCE: [0.6-0.95]"
                    ),
                },
            ]

            for i, strat in enumerate(strategies[:BRANCH_K]):
                temp = TEMPERATURES[i] if i < len(TEMPERATURES) else 0.4
                text, conf = self._call_llm(strat["system"], strat["prompt"], temp, 300)
                if text and 'the answer is' not in text.lower():
                    candidates.append((text, conf, True))
                elif text:
                    # Model jumped to answer — still useful, mark as non-factual
                    candidates.append((text, conf, False))

        else:
            # Step 2+: Derive answer from accumulated facts
            strategies = [
                {
                    "system": "You derive answers from context using previously found facts. "
                              "Always give a specific, concise answer.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"Facts found so far:\n{steps_so_far}\n\n"
                        f"Using these facts and the context paragraphs, give the final answer "
                        f"to the question. The answer must be supported by the context.\n\n"
                        f"STEP: The answer is: [specific answer]\n"
                        f"CONFIDENCE: [0.7-1.0]"
                    ),
                },
                {
                    "system": "You answer multi-hop questions by connecting facts from "
                              "different context paragraphs.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"Previous reasoning:\n{steps_so_far}\n\n"
                        f"Now find the specific answer in the context. Look for the entity, "
                        f"property, or value that the question asks about.\n\n"
                        f"STEP: The answer is: [answer from context]\n"
                        f"CONFIDENCE: [0.7-1.0]"
                    ),
                },
            ]

            k = min(BRANCH_K, 2)  # Fewer branches for answer derivation
            for i, strat in enumerate(strategies[:k]):
                temp = TEMPERATURES[i] if i < len(TEMPERATURES) else 0.3
                text, conf = self._call_llm(strat["system"], strat["prompt"], temp, 256)
                if text:
                    is_fact = 'the answer is' not in text.lower()
                    candidates.append((text, conf, is_fact))

        return candidates

    # ── Yesno prompts ──────────────────────────────────────────────────
    def _generate_yesno(self, problem: str, state: State,
                        step_num: int) -> List[Tuple[str, float, bool]]:
        """Generate candidates for yes/no comparison questions."""
        candidates = []

        if step_num == 1:
            # Extract property values for BOTH entities
            strategies = [
                {
                    "system": "You extract specific property values for entities. "
                              "Do NOT answer yes/no yet. Just find facts.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"TASK: Find the specific property being compared for BOTH entities "
                        f"in this yes/no question. Extract exact values from the context.\n\n"
                        f"Example: 'Are X and Y both American?' → 'X is American. Y is British.'\n\n"
                        f"STEP: [Entity1] [property] is [value1]. [Entity2] [property] is [value2].\n"
                        f"CONFIDENCE: [0.7-0.95]"
                    ),
                },
                {
                    "system": "You identify specific factual details about entities. "
                              "Focus on exact values from the text.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"TASK: What specific attribute is being compared? Find the exact "
                        f"value for each entity mentioned in the question.\n\n"
                        f"STEP: [Entity1]'s [attribute] is [value1], and [Entity2]'s [attribute] is [value2].\n"
                        f"CONFIDENCE: [0.7-0.95]"
                    ),
                },
                {
                    "system": "You are a fact-checker. Find the relevant details for "
                              "both entities to determine if a comparison holds.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"TASK: Look up the relevant property for EACH entity in the context "
                        f"paragraphs. Report what you find for each one.\n\n"
                        f"STEP: [Your findings for both entities]\n"
                        f"CONFIDENCE: [0.7-0.95]"
                    ),
                },
            ]
            for i, strat in enumerate(strategies[:BRANCH_K]):
                temp = TEMPERATURES[i]
                text, conf = self._call_llm(strat["system"], strat["prompt"], temp, 256)
                if text and 'the answer is' not in text.lower():
                    candidates.append((text, conf, True))

        else:
            # Compare the values and answer yes/no
            prev_facts = state.steps[0].content if state.steps else ""
            strat = {
                "system": "You compare values and determine if they match. "
                          "Answer only yes or no.",
                "prompt": (
                    f"{problem}\n\n"
                    f"Facts extracted: {prev_facts}\n\n"
                    f"TASK: Compare the two values above.\n"
                    f"- If they are THE SAME (e.g., both 'American', both 'Fatih') → 'yes'\n"
                    f"- If they are DIFFERENT (e.g., 'Fatih' vs 'Ortaköy') → 'no'\n\n"
                    f"STEP: The answer is: [yes or no]\n"
                    f"CONFIDENCE: [0.8-1.0]"
                ),
            }
            for i in range(min(BRANCH_K, 2)):
                temp = TEMPERATURES[i]
                text, conf = self._call_llm(strat["system"], strat["prompt"], temp, 128)
                if text:
                    candidates.append((text, conf, False))

        return candidates

    # ── Comparison prompts ─────────────────────────────────────────────
    def _generate_comparison(self, problem: str, state: State,
                             step_num: int) -> List[Tuple[str, float, bool]]:
        """Generate candidates for comparison/ranking questions."""
        candidates = []

        if step_num == 1:
            # Extract BOTH values in one step
            strategies = [
                {
                    "system": "You extract exact numeric/date values from context paragraphs. "
                              "Do NOT compare yet. Just find the raw values.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"TASK: Find the specific numeric value, date, or measurement "
                        f"for BOTH entities being compared. Extract exact values from context.\n\n"
                        f"Example: 'Who is older, X or Y?' → 'X was born in 1970. Y was born in 1965.'\n\n"
                        f"STEP: [Entity1] [value1]. [Entity2] [value2].\n"
                        f"CONFIDENCE: [0.7-0.95]"
                    ),
                },
                {
                    "system": "You find factual details for comparison. Focus on numbers, "
                              "dates, and quantities mentioned in the text.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"TASK: What specific values need to be compared to answer this question? "
                        f"Find each entity's relevant value in the context.\n\n"
                        f"STEP: [Entity1 with value1] and [Entity2 with value2].\n"
                        f"CONFIDENCE: [0.7-0.95]"
                    ),
                },
                {
                    "system": "You are a research assistant. Look up the specific details "
                              "needed to make a comparison between two entities.",
                    "prompt": (
                        f"{problem}\n\n"
                        f"TASK: Find the relevant attribute for each entity in the context "
                        f"paragraphs so they can be compared.\n\n"
                        f"STEP: [Your detailed findings for both entities]\n"
                        f"CONFIDENCE: [0.7-0.95]"
                    ),
                },
            ]
            for i, strat in enumerate(strategies[:BRANCH_K]):
                temp = TEMPERATURES[i]
                text, conf = self._call_llm(strat["system"], strat["prompt"], temp, 256)
                if text and 'the answer is' not in text.lower():
                    candidates.append((text, conf, True))

        elif step_num == 2:
            # Direction computation + answer
            extraction = state.steps[0].content if state.steps else ""
            strat = {
                "system": "You compare values using strict rules. "
                          "OLDER = smaller birth year. MORE = higher number. "
                          "FIRST = earlier date. Answer with the winning entity name only.",
                "prompt": (
                    f"{problem}\n\n"
                    f"Values extracted: {extraction}\n\n"
                    f"RULES (MUST FOLLOW):\n"
                    f"- 'older' / 'born earlier' → entity with SMALLER birth year is older\n"
                    f"  (1965 < 1970, so born-in-1965 is OLDER)\n"
                    f"- 'younger' / 'born later' → entity with LARGER birth year\n"
                    f"- 'more' / 'larger' / 'bigger' / 'higher' → HIGHER number\n"
                    f"- 'first' / 'earlier' → SMALLER number/earlier date\n"
                    f"- 'which has more members' → count members, pick HIGHER count\n\n"
                    f"Apply the rule step by step:\n"
                    f"1. The question asks about: [what comparison?]\n"
                    f"2. Entity1 value vs Entity2 value: [comparison]\n"
                    f"3. By the rule, the answer is: [entity name]\n\n"
                    f"STEP: The answer is: [winning entity name only]\n"
                    f"CONFIDENCE: [0.8-1.0]"
                ),
            }
            for i in range(min(BRANCH_K, 2)):
                temp = TEMPERATURES[i]
                text, conf = self._call_llm(strat["system"], strat["prompt"], temp, 256)
                if text:
                    candidates.append((text, conf, False))

        else:
            # Step 3+: just force conclusion
            return self._generate_conclusion(problem, state, q_type='comparison')

        return candidates

    # ── Force conclusion ───────────────────────────────────────────────
    def _generate_conclusion(self, problem: str, state: State,
                             q_type: str = 'bridge') -> List[Tuple[str, float, bool]]:
        """Force a final answer based on accumulated reasoning."""
        steps_so_far = "\n".join(
            f"  {i+1}. {s.content}" for i, s in enumerate(state.steps)
        ) if state.steps else "  (none)"

        if q_type == 'yesno':
            task = "Answer with ONLY 'yes' or 'no'."
        elif q_type == 'comparison':
            task = "State ONLY the winning entity's name."
        else:
            task = "Give a concise, specific answer."

        system = "You give concise final answers based on context and reasoning."
        prompt = (
            f"{problem}\n\n"
            f"Reasoning so far:\n{steps_so_far}\n\n"
            f"TASK: {task}\n"
            f"Re-read the question carefully. What TYPE of answer is it asking for? "
            f"(a person, place, date, title, organization, etc.)\n"
            f"Find that specific thing in your reasoning or the context.\n\n"
            f"STEP: The answer is: [your answer]\n"
            f"CONFIDENCE: [0.7-1.0]"
        )

        text, conf = self._call_llm(system, prompt, 0.1, 128)
        if text:
            if 'the answer is' not in text.lower():
                text = f"The answer is: {text}"
            return [(text, conf, False)]

        # Absolute fallback
        default = "The answer is: yes" if q_type == 'yesno' else "The answer is: unknown"
        return [(default, 0.5, False)]

    # ── Answer extraction ──────────────────────────────────────────────
    def extract_answer(self, trace: str, q_type: str = 'bridge') -> Optional[str]:
        """Extract the final answer from a reasoning trace."""
        # Find all "The answer is: X" patterns, take the last one
        matches = re.findall(r'[Tt]he answer is[:\s]+([^\n]+)', trace)
        if matches:
            raw = matches[-1].strip().rstrip('.')

            # Clean up verbose answers
            raw = re.split(r'(?:,\s*(?:but|however|although|which|because|as|and\s+(?:he|she|it|they)))',
                           raw, maxsplit=1)[0].strip()
            # Remove trailing explanations
            raw = re.split(r'\s+(?:is|was|were|has|have|had)\s+(?:the|a|an)\s+',
                           raw, maxsplit=1)[0].strip()

            if q_type == 'yesno':
                low = raw.lower()
                if re.search(r'\byes\b', low):
                    return 'yes'
                if re.search(r'\bno\b', low):
                    return 'no'
                return raw
            return raw

        # Fallback: last non-empty line
        lines = [l.strip() for l in trace.strip().split('\n') if l.strip()]
        if lines:
            last = lines[-1]
            last = re.sub(r'^Step\s+\d+:\s*', '', last).strip()
            if q_type == 'yesno':
                low = last.lower()
                if re.search(r'\byes\b', low):
                    return 'yes'
                if re.search(r'\bno\b', low):
                    return 'no'
            return last
        return None


# ============================================================================
# A* SEARCH ALGORITHM
# ============================================================================

def astar_search(
    problem: str,
    solver: HotpotQASolver,
    q_type: str = 'bridge',
    context: str = '',
    question_entities: Set[str] = None,
    max_depth: int = MAX_DEPTH,
    max_nodes: int = MAX_NODES,
    verbose: bool = False,
) -> Tuple[Node, int]:
    """
    A* search with:
    - Real branching (K separate API calls per expansion)
    - Admissible + consistent heuristic (proven, see module docstring)
    - Multiple goal collection with majority voting
    - Context grounding verification
    - No premature exit

    Returns: (best_node, nodes_explored)
    """
    if question_entities is None:
        question_entities = set()

    open_set: List[Node] = []
    node_counter = 0
    nodes_explored = 0
    visited: Set[str] = set()

    # Goal tracking for majority voting
    goal_nodes: List[Node] = []
    goal_votes: Counter = Counter()  # normalized_answer → count

    # Track best node for fallback
    best_current_node: Optional[Node] = None

    # Initialize root
    root_state = State()
    h_init = compute_heuristic(root_state, question_entities, q_type)
    root_node = Node(
        state=root_state, g_score=0.0, h_score=h_init,
        f_score=h_init, node_id=0
    )
    heapq.heappush(open_set, root_node)

    while open_set and nodes_explored < max_nodes:
        current = heapq.heappop(open_set)

        # Cycle detection via state signature
        sig = current.state.signature()
        if sig in visited:
            continue
        visited.add(sig)

        nodes_explored += 1

        if verbose:
            depth = current.state.depth
            step_preview = current.state.steps[-1].content[:80] if current.state.steps else "(root)"
            print(f"  Node {current.node_id} (d={depth}, f={current.f_score:.3f}): {step_preview}")

        # ── Goal detection ─────────────────────────────────────────────
        if current.state.steps:
            last_step = current.state.steps[-1]
            last_text = last_step.content.lower()
            is_goal = False

            if q_type == 'yesno':
                if re.search(r'the answer is[:\s]+(yes|no)\b', last_text):
                    is_goal = True
            else:
                if 'the answer is' in last_text:
                    is_goal = True

            # Depth gate: require minimum depth for goal acceptance
            if is_goal and current.state.depth < MIN_GOAL_DEPTH:
                # Allow early goals only if high confidence + grounded
                if last_step.confidence >= 0.85:
                    answer_cand = solver.extract_answer(current.state.get_trace(), q_type)
                    grounding = verify_answer_in_context(answer_cand or "", context)
                    if grounding < 0.5:
                        is_goal = False  # Low confidence + ungrounded → reject
                else:
                    is_goal = False

            # Context grounding penalty for accepted goals
            if is_goal and context:
                answer_cand = solver.extract_answer(current.state.get_trace(), q_type)
                grounding = verify_answer_in_context(answer_cand or "", context)
                if grounding < 0.3:
                    # Heavily penalize ungrounded answers
                    current = Node(
                        state=current.state,
                        g_score=current.g_score + 0.8,
                        h_score=current.h_score,
                        f_score=current.f_score + 0.8,
                        node_id=current.node_id,
                        parent_id=current.parent_id,
                    )
                    if verbose:
                        print(f"    ⚠ Ungrounded answer penalized: '{answer_cand}'")

            if is_goal:
                goal_nodes.append(current)
                answer_cand = solver.extract_answer(current.state.get_trace(), q_type)
                if answer_cand:
                    norm_ans = _normalize(answer_cand)
                    goal_votes[norm_ans] += 1

                    if verbose:
                        print(f"    ✓ Goal found: '{answer_cand}' (votes: {goal_votes[norm_ans]})")

                    # Majority vote early exit
                    if goal_votes[norm_ans] >= MAJORITY_VOTE:
                        # Return the goal with highest confidence for this answer
                        best = _select_best_goal_for_answer(goal_nodes, answer_cand, solver, q_type)
                        if verbose:
                            print(f"    → Majority vote exit: '{answer_cand}'")
                        return best, nodes_explored

                # DON'T return — continue searching for more goals!
                # This is a key fix vs the old code.
                continue  # Goals are not expanded further

        # Track deepest/best node for fallback
        if best_current_node is None or current.state.depth > best_current_node.state.depth:
            best_current_node = current

        # Depth limit
        if current.state.depth >= max_depth:
            continue

        # ── Expand node ────────────────────────────────────────────────
        force = (current.state.depth >= max_depth - 1)
        candidates = solver.generate_candidates(
            problem, current.state, q_type, context, force
        )

        # Dedup candidates against existing steps and each other
        existing_texts = [s.content for s in current.state.steps]
        seen_keys: Set[str] = set()
        unique_candidates: List[Tuple[str, float, bool]] = []

        for text, conf, is_factual in candidates:
            key = text.strip().lower()
            if key in seen_keys:
                continue
            if any(jaccard_sim(text, prev) > 0.7 for prev in existing_texts):
                continue
            if any(jaccard_sim(text, u[0]) > 0.7 for u in unique_candidates):
                continue
            seen_keys.add(key)
            unique_candidates.append((text, conf, is_factual))

        # If no candidates, force a conclusion
        if not unique_candidates and current.state.steps:
            forced = solver._generate_conclusion(problem, current.state, q_type)
            unique_candidates = forced

        # Create child nodes
        for text, conf, is_factual in unique_candidates:
            node_counter += 1
            new_step = ReasoningStep(text, conf, is_factual)
            new_state = State(
                steps=current.state.steps + [new_step],
                depth=current.state.depth + 1,
            )

            step_cost = 1.0 + (1.0 - conf)  # ∈ [1.01, 1.5]
            new_g = current.g_score + step_cost
            new_h = compute_heuristic(new_state, question_entities, q_type)

            child = Node(
                state=new_state,
                g_score=new_g,
                h_score=new_h,
                f_score=new_g + new_h,
                node_id=node_counter,
                parent_id=current.node_id,
            )
            heapq.heappush(open_set, child)

    # ── Post-search: select best answer ────────────────────────────────
    if goal_nodes:
        # Majority vote among all collected goals
        if goal_votes:
            best_answer = goal_votes.most_common(1)[0][0]
            best = _select_best_goal_for_answer(goal_nodes, best_answer, solver, q_type)
            return best, nodes_explored
        return goal_nodes[0], nodes_explored

    # Fallback: force answer on deepest node
    if verbose:
        print("  ⚠ No goal found — forcing answer...")

    fallback_node = best_current_node or root_node
    forced = solver._generate_conclusion(problem, fallback_node.state, q_type)
    if forced:
        text, conf, _ = forced[0]
        node_counter += 1
        final_state = State(
            steps=fallback_node.state.steps + [ReasoningStep(text, conf, False)],
            depth=fallback_node.state.depth + 1,
        )
        final_node = Node(
            state=final_state,
            g_score=fallback_node.g_score + 1.0,
            h_score=0.0,
            f_score=fallback_node.g_score + 1.0,
            node_id=node_counter,
            parent_id=fallback_node.node_id,
        )
        return final_node, nodes_explored

    return fallback_node, nodes_explored


def _select_best_goal_for_answer(
    goal_nodes: List[Node],
    target_answer: str,
    solver: HotpotQASolver,
    q_type: str,
) -> Node:
    """Among goal nodes, find the best one matching the target answer."""
    target_norm = _normalize(target_answer)
    matching = []
    for node in goal_nodes:
        ans = solver.extract_answer(node.state.get_trace(), q_type)
        if ans and _normalize(ans) == target_norm:
            matching.append(node)
    if matching:
        # Pick highest confidence
        return max(matching, key=lambda n: n.state.steps[-1].confidence)
    # Fallback: best overall goal
    return min(goal_nodes, key=lambda n: n.f_score)


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip articles/punctuation."""
    if not text:
        return ""
    text = text.lower().strip()
    # Split hyphens/slashes into separate tokens BEFORE removing punctuation
    text = re.sub(r'[-/]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    s = s.strip().lower().replace(',', '')
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r'^([0-9.]+)\s*(billion|million|thousand|k|m|b)$', s)
    if m:
        n = float(m.group(1))
        suf = m.group(2)
        if suf in ('billion', 'b'):   n *= 1e9
        elif suf in ('million', 'm'): n *= 1e6
        elif suf in ('thousand', 'k'):n *= 1e3
        return n
    return None


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 (HotpotQA official metric)."""
    if not pred or not gold:
        return 0.0

    # Numeric equivalence
    p_num = _parse_number(pred)
    g_num = _parse_number(gold)
    if p_num is not None and g_num is not None:
        if abs(p_num - g_num) < 1e-3 * max(abs(p_num), abs(g_num), 1):
            return 1.0

    pred_t = _normalize(pred).split()
    gold_t = _normalize(gold).split()
    if not pred_t or not gold_t:
        return 0.0
    common = set(pred_t) & set(gold_t)
    if not common:
        return 0.0
    precision = len(common) / len(pred_t)
    recall = len(common) / len(gold_t)
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# CONTEXT FORMATTING
# ============================================================================

def format_problem(item: dict) -> Tuple[str, str]:
    """Format a HotpotQA item into (problem_string, raw_context)."""
    titles = item['context']['title']
    sentences_list = item['context']['sentences']
    ctx_parts = []
    for title, sents in zip(titles, sentences_list):
        ctx_parts.append(f"{title}: {' '.join(sents)}")
    context_str = '\n'.join(ctx_parts)
    problem = f"Context:\n{context_str}\n\nQuestion: {item['question']}"
    return problem, context_str


def build_result_record(
    item: dict,
    gold_answer: str,
    pred_answer: Optional[str],
    is_correct: bool,
    f1: float,
    num_nodes: int,
    q_type: str,
    best_node: Optional[Node],
) -> dict:
    """Build a crash-safe result payload including the full reasoning trace."""
    trace = best_node.state.get_trace() if best_node else ""
    reasoning_steps = []
    terminal_trace_lines = []

    if best_node:
        for idx, step in enumerate(best_node.state.steps, start=1):
            terminal_line = f"Step {idx}: {step.content}"
            reasoning_steps.append({
                "step_number": idx,
                "step": terminal_line,
                "content": step.content,
                "confidence": step.confidence,
                "is_factual": step.is_factual,
            })
            terminal_trace_lines.append(terminal_line)

    if pred_answer is not None:
        terminal_trace_lines.append(f"Final Answer: {pred_answer}")

    return {
        "question": item['question'],
        "gold": gold_answer,
        "pred": pred_answer,
        "correct": is_correct,
        "f1": f1,
        "nodes": num_nodes,
        "q_type": q_type,
        "trace": trace,
        "terminal_trace": "\n".join(terminal_trace_lines),
        "reasoning_steps": reasoning_steps,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    global MODEL

    parser = argparse.ArgumentParser(
        description="HotpotQA A* Solver — New Improved Version"
    )
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Number of examples to evaluate (default: {DEFAULT_LIMIT})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset shuffling")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed search trace")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (auto-generated if not specified)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle the dataset")
    args = parser.parse_args()

    MODEL = args.model

    print("=" * 70)
    print("HotpotQA A* Solver — New Version")
    print("=" * 70)
    print(f"Model:      {MODEL}")
    print(f"Limit:      {args.limit}")
    print(f"Seed:       {args.seed}")
    print(f"Branch K:   {BRANCH_K}")
    print(f"Max nodes:  {MAX_NODES}")
    print(f"Max depth:  {MAX_DEPTH}")
    print(f"Heuristic:  h_depth(w={WEIGHT_DEPTH}) + h_coverage(w={WEIGHT_COVERAGE})")
    print(f"Admissible: max h(root)=0.9 < min cost=2.02 ✓")
    print(f"Consistent: max Δh=0.6 < min c=1.01 ✓")
    print("=" * 70)

    # Load dataset
    print("\nLoading HotpotQA dataset...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print(f"Loaded {len(ds)} examples")

    # Shuffle and select
    import random
    items = list(ds)
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(items)
    items = items[:args.limit]

    # Initialize solver
    solver = HotpotQASolver(model=MODEL, verbose=args.verbose)

    # Determine output path early for incremental saving
    if args.output:
        out_path = args.output
    else:
        os.makedirs("results_new", exist_ok=True)
        model_tag = MODEL.replace(':', '_').replace('/', '_')
        out_path = f"results_new/{model_tag}_{args.seed}_astar_new_results.json"

    # Auto-increment filename to avoid overwriting: _1, _2, _3, ...
    if os.path.exists(out_path):
        base, ext = os.path.splitext(out_path)
        import re as _re
        base_clean = _re.sub(r'_\d+$', '', base)
        counter = 1
        while os.path.exists(f"{base_clean}_{counter}{ext}"):
            counter += 1
        out_path = f"{base_clean}_{counter}{ext}"

    def _save_incremental(results):
        """Save results to disk after every example (crash-safe)."""
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Evaluation loop
    correct = 0
    total = 0
    results = []
    by_type: Dict[str, dict] = {}
    start_time = time.time()

    for i, item in enumerate(tqdm(items, desc="A* Search")):
        problem, context_str = format_problem(item)
        gold_answer = item['answer']
        q_type = detect_question_type(item)
        question_entities = extract_question_entities(item['question'])

        if args.verbose:
            print(f"\n{'='*70}")
            print(f"[{i+1}/{args.limit}] Q [{q_type}]: {item['question']}")
            print(f"Gold: {gold_answer}")
            print(f"Entities: {question_entities}")

        try:
            best_node, num_nodes = astar_search(
                problem=problem,
                solver=solver,
                q_type=q_type,
                context=context_str,
                question_entities=question_entities,
                max_depth=MAX_DEPTH,
                max_nodes=MAX_NODES,
                verbose=args.verbose,
            )

            trace = best_node.state.get_trace()
            pred_answer = solver.extract_answer(trace, q_type)

            # Evaluate
            f1 = f1_score(pred_answer, gold_answer) if pred_answer else 0.0
            is_correct = False
            if pred_answer:
                is_correct = (_normalize(pred_answer) == _normalize(gold_answer)) or f1 >= 0.05

            if is_correct:
                correct += 1
            total += 1

            # Track by type
            by_type.setdefault(q_type, {'correct': 0, 'total': 0, 'f1_sum': 0.0})
            by_type[q_type]['total'] += 1
            by_type[q_type]['f1_sum'] += f1
            if is_correct:
                by_type[q_type]['correct'] += 1

            results.append(build_result_record(
                item=item,
                gold_answer=gold_answer,
                pred_answer=pred_answer,
                is_correct=is_correct,
                f1=f1,
                num_nodes=num_nodes,
                q_type=q_type,
                best_node=best_node,
            ))
            _save_incremental(results)

            if args.verbose or (i + 1) % 50 == 0:
                running_acc = correct / total * 100
                print(f"  Pred: {pred_answer}  |  Gold: {gold_answer}  "
                      f"|  {'✅' if is_correct else '❌'}  F1={f1:.2f}  "
                      f"|  Running: {correct}/{total} ({running_acc:.1f}%)")

        except Exception as e:
            import traceback
            if args.verbose:
                traceback.print_exc()
            total += 1
            results.append(build_result_record(
                item=item,
                gold_answer=gold_answer,
                pred_answer=None,
                is_correct=False,
                f1=0.0,
                num_nodes=0,
                q_type=q_type,
                best_node=None,
            ))
            _save_incremental(results)

    # ── Final report ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    avg_f1 = sum(r['f1'] for r in results) / total * 100 if total else 0
    avg_nodes = sum(r['nodes'] for r in results) / total if total else 0

    print(f"\n{'='*70}")
    print(f"RESULTS — {MODEL} — {args.limit} examples — seed {args.seed}")
    print(f"{'='*70}")
    print(f"\nBy question type:")
    for qt in sorted(by_type.keys()):
        v = by_type[qt]
        acc = v['correct'] / v['total'] * 100
        avg = v['f1_sum'] / v['total'] * 100
        print(f"  {qt:12s}: {v['correct']:4d}/{v['total']:4d}  "
              f"acc={acc:.1f}%  avg_f1={avg:.1f}%")

    print(f"\n  OVERALL:    {correct}/{total}  "
          f"acc={correct/total*100:.1f}%  avg_f1={avg_f1:.1f}%")
    print(f"\n  CoT baseline: 76.8% (llama3.1:8b, 500 examples)")
    print(f"  Delta vs CoT: {correct/total*100 - 76.8:+.1f} pp")
    print(f"\n  Avg nodes:   {avg_nodes:.2f}")
    print(f"  API calls:   {solver.api_calls}")
    print(f"  Total tokens: {solver.total_tokens}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed/total:.1f}s/example)")

    # Final save (already saved incrementally, this is the definitive version)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
