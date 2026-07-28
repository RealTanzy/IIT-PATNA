#!/usr/bin/env python3
"""
StrategyQA A* Search Solver
============================

Based on HotpotQA A* solver (logicqa_new/hotpotqa_astar_new.py).

KEY DIFFERENCES vs HotpotQA:
──────────────────────────────
1. ALL questions are YES/NO — no bridge/comparison types.
2. NO context passages — model must reason from world knowledge.
3. Gold answer is boolean (True/False → "yes"/"no").
4. Prompts are adapted for commonsense/factual world-knowledge reasoning.
5. Grounding check is against the model's own reasoning trace (no passage).

ADMISSIBILITY (same proof structure, simplified for 2-hop yesno):
─────────────────────────────────────────────────────────────────
All StrategyQA questions require at least 2 reasoning steps:
  Step 1: Identify the relevant facts (what do we need to know?)
  Step 2: Conclude yes/no based on those facts.

h(s) = h_depth(s) + h_coverage(s)
  h_depth(s)    = max(0, 2 - depth(s)) × 0.3
  h_coverage(s) = (1 - entity_coverage) × 0.3

Step cost: c(s→s') = 1.0 + (1-confidence) ≥ 1.01
Min path:  2 steps × 1.01 = 2.02
max h(root) = 2×0.3 + 1×0.3 = 0.9 < 2.02  ✓ (admissible)
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
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "llama3.2:3b"
DEFAULT_LIMIT = 500
BASE_URL = "http://localhost:11434/v1"

# Heuristic weights (admissible — see module docstring)
WEIGHT_DEPTH    = 0.3
WEIGHT_COVERAGE = 0.3

# Search parameters
BRANCH_K       = 2      # Candidates per expansion
MAX_DEPTH      = 8      # Maximum reasoning depth
MAX_NODES      = 15     # Maximum nodes to explore
MIN_GOAL_DEPTH = 2      # Minimum depth before accepting a goal
MAJORITY_VOTE  = 2      # Votes needed for early exit
TEMPERATURES   = [0.2, 0.6]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8
    is_factual: bool = False  # True: fact extraction; False: conclusion

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
    g_score: float
    h_score: float
    f_score: float
    node_id: int
    parent_id: Optional[int] = None

    def __lt__(self, other: "Node") -> bool:
        if abs(self.f_score - other.f_score) > 1e-9:
            return self.f_score < other.f_score
        return self.node_id < other.node_id


# ============================================================================
# ENTITY EXTRACTION (for heuristic)
# ============================================================================

_STOPWORDS = frozenset({
    'a', 'an', 'the', 'is', 'was', 'were', 'are', 'of', 'in', 'on', 'at',
    'for', 'and', 'or', 'to', 'by', 'with', 'it', 'its', 'that', 'this',
    'from', 'as', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
    'what', 'which', 'who', 'whom', 'where', 'when', 'how', 'why',
    'same', 'both', 'also', 'not', 'no', 'yes', 'than', 'more', 'most',
    'other', 'some', 'any', 'all', 'many', 'much', 'few', 'several',
    'first', 'last', 'new', 'old', 'older', 'younger',
})


def extract_question_entities(question: str) -> Set[str]:
    """Extract meaningful entities from a question for heuristic evaluation."""
    entities = set()

    # Quoted strings
    for m in re.finditer(r'"([^"]+)"', question):
        entities.add(m.group(1).lower())

    # Capitalized phrases (proper nouns)
    q_trimmed = re.sub(
        r'^(What|Which|Who|Where|When|How|Are|Were|Is|Was|Did|Do|Does|Could|Would|Can|Has|Have|Had|The)\s+',
        '', question, flags=re.IGNORECASE
    )
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', q_trimmed):
        phrase = m.group(1).lower()
        if phrase not in _STOPWORDS and len(phrase) > 2:
            entities.add(phrase)

    # Numbers / years
    for m in re.finditer(r'\b(\d{4}|\d+(?:,\d{3})*)\b', question):
        entities.add(m.group(1))

    # Key content words
    _GENERIC = _STOPWORDS | frozenset({
        'person', 'people', 'human', 'animal', 'thing', 'place', 'object',
        'type', 'kind', 'part', 'year', 'date', 'time', 'name', 'known',
        'called', 'named', 'located', 'founded', 'formed', 'able', 'capable',
        'possible', 'likely', 'usually', 'generally', 'often', 'typically',
    })
    words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
    for w in words:
        if w not in _GENERIC and len(w) > 4:
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

def compute_heuristic(state: State, question_entities: Set[str]) -> float:
    """
    h(s) = h_depth(s) + h_coverage(s)
    All StrategyQA questions need ≥2 hops → same admissibility argument as HotpotQA.
    """
    remaining_hops = max(0, 2 - state.depth)
    h_depth = remaining_hops * WEIGHT_DEPTH

    coverage = compute_entity_coverage(state, question_entities)
    h_coverage = (1.0 - coverage) * WEIGHT_COVERAGE

    return h_depth + h_coverage


# ============================================================================
# SIMILARITY HELPERS
# ============================================================================

def jaccard_sim(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _is_degenerate(content: str) -> bool:
    """Detect repetitive/degenerate LLM output."""
    if len(content) < 10:
        return False
    stripped = content.strip()
    if len(set(stripped)) <= 3 and len(stripped) > 20:
        return True
    words = stripped.split()
    if len(words) > 10:
        most_common = Counter(words).most_common(1)[0]
        if most_common[1] / len(words) > 0.6:
            return True
    return False


# ============================================================================
# SOLVER
# ============================================================================

class StrategyQASolver:
    """
    LLM-backed A* solver for StrategyQA (yes/no, no context).
    Reasoning is done purely from world knowledge.
    """

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
        """Make a single LLM call. Returns (step_text, confidence)."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["\n\n\n"],
            )
            self.total_tokens += resp.usage.total_tokens
            self.api_calls += 1
            content = resp.choices[0].message.content or ""

            if _is_degenerate(content):
                return None, 0.0

            return self._parse_output(content)
        except Exception as e:
            if self.verbose:
                print(f"    LLM error: {e}")
            return None, 0.0

    def _parse_output(self, content: str) -> Tuple[Optional[str], float]:
        """Parse LLM output robustly."""
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

        # "The answer is: ..." fallback
        ans_m = re.search(r'[Tt]he answer is[:\s]+(.+?)(?:\.|$)', content)
        if ans_m:
            text = f"The answer is: {ans_m.group(1).strip()}"
            conf = float(conf_m.group(1)) if conf_m else 0.75
            return text, max(0.5, min(0.99, conf))

        # Last resort: use entire response
        text = content.strip()
        text = re.sub(r'^(?:Step\s*\d+[:\s]*|STEP[:\s]*)', '', text, flags=re.IGNORECASE).strip()
        if 5 < len(text) < 500:
            return text, 0.6

        return None, 0.0

    # ── Step 1: Fact gathering (world knowledge) ───────────────────────
    def _generate_fact_step(self, question: str, state: State,
                            step_num: int) -> List[Tuple[str, float, bool]]:
        """
        Generate intermediate reasoning steps from world knowledge.
        Step 1: identify what facts are needed and recall them.
        """
        candidates = []
        steps_so_far = "\n".join(
            f"  {i+1}. {s.content}" for i, s in enumerate(state.steps)
        ) if state.steps else ""

        if step_num == 1:
            strategies = [
                # Strategy A: Decompose then recall
                {
                    "system": (
                        "You are a knowledgeable assistant that breaks down yes/no questions. "
                        "First identify the key sub-facts needed, then recall them from memory. "
                        "Do NOT give the yes/no answer yet — only gather facts."
                    ),
                    "prompt": (
                        f"Question: {question}\n\n"
                        f"TASK: What intermediate facts do you need to answer this question? "
                        f"Recall 1-2 specific facts from your knowledge.\n\n"
                        f"Example: 'Could a penguin outrun a human?' → "
                        f"'Penguins waddle at ~2.5 mph. Average human walks at 3.5 mph. "
                        f"Sprinting humans reach 15+ mph.'\n\n"
                        f"STEP: [The key facts relevant to this question]\n"
                        f"CONFIDENCE: [0.6-0.95]"
                    ),
                },
                # Strategy B: Entity-attribute lookup
                {
                    "system": (
                        "You recall factual information about entities from your world knowledge. "
                        "Be specific and concise. Do not answer yes or no yet."
                    ),
                    "prompt": (
                        f"Question: {question}\n\n"
                        f"TASK: Identify the entities or concepts in this question and "
                        f"recall the relevant property or fact for each one.\n\n"
                        f"STEP: [Entity/Concept 1]: [relevant fact]. [Entity/Concept 2]: [relevant fact].\n"
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
                    candidates.append((text, conf, False))

        else:
            # Step 2+: draw conclusion from gathered facts
            strat = {
                "system": (
                    "You answer yes/no questions using previously gathered facts. "
                    "Answer ONLY 'yes' or 'no'."
                ),
                "prompt": (
                    f"Question: {question}\n\n"
                    f"Facts gathered so far:\n{steps_so_far}\n\n"
                    f"TASK: Based on the facts above, answer the question.\n"
                    f"- If the facts support the claim → 'yes'\n"
                    f"- If the facts contradict the claim → 'no'\n\n"
                    f"STEP: The answer is: [yes or no]\n"
                    f"CONFIDENCE: [0.7-1.0]"
                ),
            }
            for i in range(min(BRANCH_K, 2)):
                temp = TEMPERATURES[i]
                text, conf = self._call_llm(strat["system"], strat["prompt"], temp, 128)
                if text:
                    candidates.append((text, conf, False))

        return candidates

    def generate_candidates(
        self,
        question: str,
        state: State,
        force_conclusion: bool = False,
    ) -> List[Tuple[str, float, bool]]:
        """Generate reasoning candidates. All StrategyQA questions are yes/no."""
        step_num = len(state.steps) + 1

        if force_conclusion:
            return self._generate_conclusion(question, state)

        return self._generate_fact_step(question, state, step_num)

    # ── Force conclusion ───────────────────────────────────────────────
    def _generate_conclusion(self, question: str,
                             state: State) -> List[Tuple[str, float, bool]]:
        """Force a yes/no conclusion based on accumulated reasoning."""
        steps_so_far = "\n".join(
            f"  {i+1}. {s.content}" for i, s in enumerate(state.steps)
        ) if state.steps else "  (none)"

        system = "You give concise yes/no answers based on reasoning."
        prompt = (
            f"Question: {question}\n\n"
            f"Reasoning so far:\n{steps_so_far}\n\n"
            f"TASK: Based on your reasoning, answer yes or no.\n"
            f"Use your world knowledge if needed.\n\n"
            f"STEP: The answer is: [yes or no]\n"
            f"CONFIDENCE: [0.7-1.0]"
        )

        text, conf = self._call_llm(system, prompt, 0.1, 128)
        if text:
            if 'the answer is' not in text.lower():
                text = f"The answer is: {text}"
            return [(text, conf, False)]

        return [("The answer is: no", 0.5, False)]

    # ── Answer extraction ──────────────────────────────────────────────
    def extract_answer(self, trace: str) -> Optional[str]:
        """Extract yes/no from reasoning trace."""
        # Find all "The answer is: X" patterns, take the last one
        matches = re.findall(r'[Tt]he answer is[:\s]+([^\n]+)', trace)
        if matches:
            raw = matches[-1].strip().lower().rstrip('.')
            raw = re.sub(r'^(?:Answer|Response|Result)[:\s]+', '', raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r'\s*\([^)]*\)\s*$', '', raw).strip()
            # Normalise yes/no
            if re.search(r'\byes\b', raw):
                return 'yes'
            if re.search(r'\bno\b', raw):
                return 'no'
            # Affirmative synonyms
            if re.search(r'\b(true|correct|indeed|affirmative)\b', raw):
                return 'yes'
            if re.search(r'\b(false|incorrect|negative)\b', raw):
                return 'no'
            return raw

        # Fallback: scan last line for yes/no
        lines = [l.strip() for l in trace.strip().split('\n') if l.strip()]
        if lines:
            last = lines[-1].lower()
            if re.search(r'\byes\b', last):
                return 'yes'
            if re.search(r'\bno\b', last):
                return 'no'
        return None


# ============================================================================
# A* SEARCH ALGORITHM
# ============================================================================

def astar_search(
    question: str,
    solver: StrategyQASolver,
    question_entities: Set[str] = None,
    max_depth: int = MAX_DEPTH,
    max_nodes: int = MAX_NODES,
    verbose: bool = False,
) -> Tuple[Node, int]:
    """
    A* search for StrategyQA.
    - Real branching via separate API calls
    - Admissible heuristic (h_depth + h_coverage)
    - No premature exit — collect multiple goals, majority vote
    """
    if question_entities is None:
        question_entities = set()

    open_set: List[Node] = []
    node_counter = 0
    nodes_explored = 0
    visited: Set[str] = set()

    goal_nodes: List[Node] = []
    goal_votes: Counter = Counter()
    best_current_node: Optional[Node] = None

    # Root node
    root_state = State()
    h_init = compute_heuristic(root_state, question_entities)
    root_node = Node(
        state=root_state, g_score=0.0, h_score=h_init,
        f_score=h_init, node_id=0
    )
    heapq.heappush(open_set, root_node)

    # ── COT seed: full reasoning attempt injected as high-priority node ──
    cot_system = (
        "You answer yes/no questions using your world knowledge. "
        "Reason step by step, then give a clear yes or no answer."
    )
    cot_prompt = (
        f"Question: {question}\n\n"
        f"Think through this step by step using your world knowledge. "
        f"End with: The answer is: [yes or no]\n"
    )
    cot_text, cot_conf = solver._call_llm(cot_system, cot_prompt, 0.2, 512)
    if cot_text:
        node_counter += 1
        has_answer = 'the answer is' in (cot_text or '').lower()
        cot_step = ReasoningStep(cot_text, cot_conf, is_factual=not has_answer)
        cot_state = State(
            steps=[cot_step],
            depth=2 if has_answer else 1,
        )
        cot_g = 1.0 + (1.0 - cot_conf)
        cot_h = compute_heuristic(cot_state, question_entities)
        cot_node = Node(
            state=cot_state, g_score=cot_g, h_score=cot_h,
            f_score=cot_g + cot_h, node_id=node_counter, parent_id=0,
        )
        heapq.heappush(open_set, cot_node)
        if verbose:
            preview = cot_text[:80]
            print(f"  COT seed (conf={cot_conf:.2f}, has_answer={has_answer}): {preview}")

    while open_set and nodes_explored < max_nodes:
        current = heapq.heappop(open_set)

        sig = current.state.signature()
        if sig in visited:
            continue
        visited.add(sig)

        nodes_explored += 1

        if verbose:
            step_preview = current.state.steps[-1].content[:80] if current.state.steps else "(root)"
            print(f"  Node {current.node_id} (d={current.state.depth}, f={current.f_score:.3f}): {step_preview}")

        # ── Goal detection ─────────────────────────────────────────────
        if current.state.steps:
            last_step = current.state.steps[-1]
            last_text = last_step.content.lower()

            is_goal = bool(re.search(r'the answer is[:\s]+(yes|no)\b', last_text))

            # Depth gate
            if is_goal and current.state.depth < MIN_GOAL_DEPTH:
                if last_step.confidence < 0.75:
                    is_goal = False

            if is_goal:
                goal_nodes.append(current)
                answer_cand = solver.extract_answer(current.state.get_trace())
                if answer_cand:
                    norm_ans = answer_cand.lower().strip()
                    goal_votes[norm_ans] += 1

                    if verbose:
                        print(f"    ✓ Goal: '{answer_cand}' (votes: {goal_votes[norm_ans]})")

                    if goal_votes[norm_ans] >= MAJORITY_VOTE:
                        best = _select_best_goal(goal_nodes, answer_cand, solver)
                        if verbose:
                            print(f"    → Majority vote exit: '{answer_cand}'")
                        return best, nodes_explored

                continue  # Don't expand goal nodes

        # Track deepest node for fallback
        if best_current_node is None or current.state.depth > best_current_node.state.depth:
            best_current_node = current

        if current.state.depth >= max_depth:
            continue

        # ── Expand ─────────────────────────────────────────────────────
        force = (current.state.depth >= max_depth - 1)
        candidates = solver.generate_candidates(question, current.state, force)

        # Dedup
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

        if not unique_candidates and current.state.steps:
            unique_candidates = solver._generate_conclusion(question, current.state)

        for text, conf, is_factual in unique_candidates:
            node_counter += 1
            new_step = ReasoningStep(text, conf, is_factual)
            new_state = State(
                steps=current.state.steps + [new_step],
                depth=current.state.depth + 1,
            )
            step_cost = 1.0 + (1.0 - conf)
            new_g = current.g_score + step_cost
            new_h = compute_heuristic(new_state, question_entities)

            child = Node(
                state=new_state, g_score=new_g, h_score=new_h,
                f_score=new_g + new_h, node_id=node_counter,
                parent_id=current.node_id,
            )
            heapq.heappush(open_set, child)

    # ── Post-search: select best goal ──────────────────────────────────
    if goal_nodes:
        if goal_votes:
            top_answer, top_count = goal_votes.most_common(1)[0]
            if top_count >= 2:
                return _select_best_goal(goal_nodes, top_answer, solver), nodes_explored
        best = min(goal_nodes, key=lambda n: n.f_score)
        return best, nodes_explored

    # Fallback: force answer on deepest node
    if verbose:
        print("  ⚠ No goal found — forcing answer...")

    fallback_node = best_current_node or root_node
    forced = solver._generate_conclusion(question, fallback_node.state)
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


def _select_best_goal(
    goal_nodes: List[Node],
    target_answer: str,
    solver: StrategyQASolver,
) -> Node:
    """Among goal nodes, pick the best one matching target_answer."""
    target_norm = target_answer.lower().strip()
    matching = [
        n for n in goal_nodes
        if (solver.extract_answer(n.state.get_trace()) or "").lower().strip() == target_norm
    ]
    if matching:
        return max(matching, key=lambda n: n.state.steps[-1].confidence)
    return min(goal_nodes, key=lambda n: n.f_score)


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    return text.lower().strip().rstrip('.')


def exact_match(pred: Optional[str], gold: str) -> bool:
    """Exact match for yes/no answers."""
    if not pred:
        return False
    return _normalize(pred) == _normalize(gold)


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_strategyqa(local_path: Optional[str] = None) -> List[dict]:
    """
    Load StrategyQA dataset.
    Priority:
      1. local_path if provided
      2. Default local JSON (strategyqa_new/strategyqa_astar_routed_questions.json)
      3. HuggingFace hub (wics/strategy-qa)

    Returns list of dicts with keys: 'question', 'answer' (str: 'yes'/'no').
    """
    # Try local JSON
    candidates = []
    if local_path:
        candidates.append(local_path)
    # Default local file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "strategyqa_astar_routed_questions.json"))

    for path in candidates:
        if os.path.exists(path):
            print(f"Loading StrategyQA from local file: {path}")
            with open(path) as f:
                raw = json.load(f)
            items = []
            for entry in raw:
                gold = entry.get("gold", entry.get("answer", entry.get("label")))
                if gold is None:
                    continue
                # Normalise boolean → yes/no string
                if isinstance(gold, bool):
                    gold_str = "yes" if gold else "no"
                else:
                    gold_str = str(gold).lower().strip()
                items.append({
                    "question": entry["question"],
                    "answer": gold_str,
                })
            print(f"Loaded {len(items)} examples")
            return items

    # Fallback: HuggingFace
    print("Trying to load StrategyQA from HuggingFace (wics/strategy-qa)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wics/strategy-qa", split="test")
        items = []
        for entry in ds:
            gold = entry.get("answer", entry.get("label"))
            if isinstance(gold, bool):
                gold_str = "yes" if gold else "no"
            else:
                gold_str = str(gold).lower().strip()
            items.append({
                "question": entry["question"],
                "answer": gold_str,
            })
        print(f"Loaded {len(items)} examples from HuggingFace")
        return items
    except Exception as e:
        print(f"HuggingFace load failed: {e}")
        raise RuntimeError(
            "Could not load StrategyQA dataset. "
            "Provide --data-path or ensure wics/strategy-qa is accessible."
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    global MODEL

    parser = argparse.ArgumentParser(
        description="StrategyQA A* Solver (yes/no, world knowledge)"
    )
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Number of examples to evaluate (default: {DEFAULT_LIMIT})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed search trace")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle the dataset")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to local StrategyQA JSON file")
    args = parser.parse_args()
    MODEL = args.model

    print("=" * 70)
    print("StrategyQA A* Solver")
    print("=" * 70)
    print(f"Model:      {MODEL}")
    print(f"Limit:      {args.limit}")
    print(f"Seed:       {args.seed}")
    print(f"Branch K:   {BRANCH_K}")
    print(f"Max nodes:  {MAX_NODES}")
    print(f"Max depth:  {MAX_DEPTH}")
    print(f"Heuristic:  h_depth(w={WEIGHT_DEPTH}) + h_coverage(w={WEIGHT_COVERAGE})")
    print("=" * 70)

    # Load dataset
    print()
    items = load_strategyqa(args.data_path)

    # Shuffle and select
    import random
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(items)
    items = items[:args.limit]

    # Init solver
    solver = StrategyQASolver(model=MODEL, verbose=args.verbose)

    # Determine output path early for incremental saving
    if args.output:
        out_path = args.output
    else:
        os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
        model_tag = MODEL.replace(':', '_').replace('/', '_')
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{model_tag}_{args.seed}_strategyqa_astar_results.json"
        )

    def _save_incremental(results, correct, total):
        """Save results to disk after every example (crash-safe)."""
        save_results = [{k: v for k, v in r.items() if k != "trace"} for r in results]
        with open(out_path, 'w') as f:
            json.dump(save_results, f, indent=2)

    # Evaluation loop
    correct = 0
    total = 0
    results = []
    start_time = time.time()

    for i, item in enumerate(tqdm(items, desc="A* Search")):
        question = item["question"]
        gold_answer = item["answer"]  # 'yes' or 'no'
        question_entities = extract_question_entities(question)

        if args.verbose:
            print(f"\n{'='*70}")
            print(f"[{i+1}/{args.limit}] Q: {question}")
            print(f"Gold: {gold_answer}")
            print(f"Entities: {question_entities}")

        try:
            best_node, num_nodes = astar_search(
                question=question,
                solver=solver,
                question_entities=question_entities,
                max_depth=MAX_DEPTH,
                max_nodes=MAX_NODES,
                verbose=args.verbose,
            )

            trace = best_node.state.get_trace()
            pred_answer = solver.extract_answer(trace)

            is_correct = exact_match(pred_answer, gold_answer)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": question,
                "gold": gold_answer,
                "pred": pred_answer,
                "correct": is_correct,
                "nodes": num_nodes,
                "trace": trace,
            })
            _save_incremental(results, correct, total)

            if args.verbose or (i + 1) % 50 == 0:
                running_acc = correct / total * 100
                print(
                    f"  Pred: {pred_answer}  |  Gold: {gold_answer}  "
                    f"|  {'✅' if is_correct else '❌'}  "
                    f"|  Running: {correct}/{total} ({running_acc:.1f}%)"
                )

        except Exception as e:
            import traceback
            if args.verbose:
                traceback.print_exc()
            total += 1
            results.append({
                "question": question,
                "gold": gold_answer,
                "pred": None,
                "correct": False,
                "nodes": 0,
                "trace": "",
            })
            _save_incremental(results, correct, total)

    # ── Final report ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    avg_nodes = sum(r["nodes"] for r in results) / total if total else 0

    print(f"\n{'='*70}")
    print(f"RESULTS — {MODEL} — {args.limit} examples — seed {args.seed}")
    print(f"{'='*70}")
    yes_r = [r for r in results if r["gold"] == "yes"]
    no_r  = [r for r in results if r["gold"] == "no"]
    yes_acc = sum(r["correct"] for r in yes_r) / len(yes_r) * 100 if yes_r else 0
    no_acc  = sum(r["correct"] for r in no_r)  / len(no_r)  * 100 if no_r  else 0

    print(f"\n  Gold=yes:  {sum(r['correct'] for r in yes_r):4d}/{len(yes_r):4d}  acc={yes_acc:.1f}%")
    print(f"  Gold=no:   {sum(r['correct'] for r in no_r):4d}/{len(no_r):4d}  acc={no_acc:.1f}%")
    print(f"\n  OVERALL:   {correct}/{total}  acc={correct/total*100:.1f}%")
    print(f"\n  Avg nodes:   {avg_nodes:.2f}")
    print(f"  API calls:   {solver.api_calls}")
    print(f"  Total tokens: {solver.total_tokens}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed/total:.1f}s/example)")

    # Final save (already saved incrementally, this is the definitive version)
    save_results = [{k: v for k, v in r.items() if k != "trace"} for r in results]
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
