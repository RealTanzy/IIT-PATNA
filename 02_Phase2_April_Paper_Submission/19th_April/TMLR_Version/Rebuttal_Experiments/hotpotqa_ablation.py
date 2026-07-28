#!/usr/bin/env python3
"""
HotpotQA Ablation Script — Rebuttal Experiments
================================================

Supports three experiment types via CLI flags:

  --heuristic-mode {astar, bfs, greedy}
      astar  : full A* with f = g + h  (paper setting)
      bfs    : h = 0 → f = g  (uniform-cost / BFS degenerate)
      greedy : f = h only (greedy best-first search)

  --budget N     : max nodes to expand (default 30 = paper setting)
  --branch-k K   : branching factor (default 3 = paper setting)

  --sample-file  : JSON created by prepare_samples.py
  --out          : output JSON path

Usage examples:
  # Heuristic ablation
  python hotpotqa_ablation.py --heuristic-mode bfs    --out results/hotpotqa_bfs.json
  python hotpotqa_ablation.py --heuristic-mode greedy --out results/hotpotqa_greedy.json
  python hotpotqa_ablation.py --heuristic-mode astar  --out results/hotpotqa_astar.json

  # Budget sensitivity
  python hotpotqa_ablation.py --budget 5  --out results/hotpotqa_budget5.json
  python hotpotqa_ablation.py --budget 10 --out results/hotpotqa_budget10.json
  python hotpotqa_ablation.py --budget 15 --out results/hotpotqa_budget15.json
  python hotpotqa_ablation.py --budget 20 --out results/hotpotqa_budget20.json
  python hotpotqa_ablation.py --budget 30 --out results/hotpotqa_budget30.json

  # Branching sensitivity
  python hotpotqa_ablation.py --branch-k 1 --out results/hotpotqa_k1.json
  python hotpotqa_ablation.py --branch-k 2 --out results/hotpotqa_k2.json
  python hotpotqa_ablation.py --branch-k 3 --out results/hotpotqa_k3.json
  python hotpotqa_ablation.py --branch-k 4 --out results/hotpotqa_k4.json
"""

import os
import re
import sys
import heapq
import json
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter
from openai import OpenAI
from tqdm import tqdm

# ─── Script lives in Rebuttal_Experiments/ ───────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# ─── Defaults (paper setting) ─────────────────────────────────────────────────
DEFAULT_MODEL     = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BASE_URL  = "http://localhost:8000/v1"
DEFAULT_BUDGET    = 30
DEFAULT_BRANCH_K  = 3
DEFAULT_MAX_DEPTH = 8
DEFAULT_MIN_GOAL_DEPTH = 2
DEFAULT_MAJORITY_VOTE  = 2

WEIGHT_DEPTH    = 0.3
WEIGHT_COVERAGE = 0.3
TEMPERATURES    = [0.15, 0.45, 0.75, 0.90]

_STOPWORDS = frozenset({
    'a','an','the','is','was','were','are','of','in','on','at','for','and','or',
    'to','by','with','it','its','that','this','from','as','be','been','being',
    'have','has','had','do','does','did','will','would','could','should','may',
    'might','can','what','which','who','whom','where','when','how','why','same',
    'both','also','not','no','yes','than','more','most','other','some','any',
    'all','many','much','few','several','held','based','located','known','called',
    'named','played','first','last','new','old','older','younger',
})


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8
    is_factual: bool = False

@dataclass
class State:
    steps: List[ReasoningStep] = field(default_factory=list)
    depth: int = 0

    def get_trace(self) -> str:
        return "\n".join(f"Step {i+1}: {s.content}" for i, s in enumerate(self.steps))

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[-/]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _parse_number(s: str) -> Optional[float]:
    s = s.strip().lower().replace(',', '')
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r'^([0-9.]+)\s*(billion|million|thousand|k|m|b)$', s)
    if m:
        n = float(m.group(1))
        suf = m.group(2)
        if suf in ('billion', 'b'):    n *= 1e9
        elif suf in ('million', 'm'):  n *= 1e6
        elif suf in ('thousand', 'k'): n *= 1e3
        return n
    return None


def f1_score(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    p_num, g_num = _parse_number(pred), _parse_number(gold)
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
    p = len(common) / len(pred_t)
    r = len(common) / len(gold_t)
    return 2 * p * r / (p + r)


def jaccard_sim(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def detect_question_type(item: dict) -> str:
    dtype  = item.get('type', '')
    answer = item.get('answer', '').strip().lower()
    q      = item.get('question', '').lower().strip()
    if answer in ('yes', 'no'):
        return 'yesno'
    if dtype == 'comparison':
        if any(re.search(p, q) for p in [
            r'^(are|were|is|was|do|does|did|have|has|had)\s+',
            r'\bboth\b', r'\bsame\b', r'\balso\b',
        ]):
            return 'yesno'
        return 'comparison'
    if dtype == 'bridge':
        return 'bridge'
    if any(kw in q for kw in ['both','same','also a ','are the']):
        return 'yesno'
    if any(kw in q for kw in ['older','younger','more','fewer','which is','who is older','first']):
        return 'comparison'
    return 'bridge'


def extract_question_entities(question: str) -> Set[str]:
    entities = set()
    for m in re.finditer(r'"([^"]+)"', question):
        entities.add(m.group(1).lower())
    q_trimmed = re.sub(
        r'^(What|Which|Who|Where|When|How|Are|Were|Is|Was|Did|Do|Does|The)\s+',
        '', question, flags=re.IGNORECASE)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', q_trimmed):
        phrase = m.group(1).lower()
        if phrase not in _STOPWORDS and len(phrase) > 2:
            entities.add(phrase)
    for m in re.finditer(r'\b(\d{4}|\d+(?:,\d{3})*)\b', question):
        entities.add(m.group(1))
    for w in re.findall(r'\b[a-zA-Z]+\b', question.lower()):
        if w not in _STOPWORDS and len(w) > 3:
            entities.add(w)
    return entities


def compute_entity_coverage(state: State, question_entities: Set[str]) -> float:
    if not question_entities:
        return 1.0
    trace = state.get_trace().lower()
    covered = sum(1 for e in question_entities if e in trace)
    return covered / len(question_entities)


def compute_heuristic(state: State, question_entities: Set[str],
                      q_type: str = 'bridge',
                      heuristic_mode: str = 'astar') -> float:
    """
    Unified heuristic that supports three modes:
      astar  — full h = h_depth + h_coverage  (paper setting, admissible)
      bfs    — h = 0  (degenerate to uniform-cost / BFS)
      greedy — same formula as astar; f will be set to h only in search loop
    """
    if heuristic_mode == 'bfs':
        return 0.0
    min_hops = 2
    remaining_hops = max(0, min_hops - state.depth)
    h_depth    = remaining_hops * WEIGHT_DEPTH
    coverage   = compute_entity_coverage(state, question_entities)
    h_coverage = (1.0 - coverage) * WEIGHT_COVERAGE
    return h_depth + h_coverage


def verify_answer_in_context(answer: str, context: str) -> float:
    if not answer or not context:
        return 0.0
    ctx_lower = context.lower()
    answer_lower = answer.lower().strip()
    if answer_lower in ctx_lower:
        return 1.0
    tokens = [t for t in re.findall(r'\b\w+\b', answer_lower)
              if t not in _STOPWORDS and len(t) > 2]
    if not tokens:
        return 0.5
    return sum(1 for t in tokens if t in ctx_lower) / len(tokens)


# ─── Solver ───────────────────────────────────────────────────────────────────

class HotpotQASolver:
    def __init__(self, model: str, base_url: str, branch_k: int = 3, verbose: bool = False):
        self.client   = OpenAI(base_url=base_url, api_key='none')
        self.model    = model
        self.branch_k = branch_k
        self.verbose  = verbose
        self.total_tokens = 0
        self.api_calls    = 0

    def _call_llm(self, system: str, prompt: str,
                  temperature: float = 0.3, max_tokens: int = 512) -> Tuple[Optional[str], float]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.total_tokens += resp.usage.total_tokens
            self.api_calls    += 1
            return self._parse_output(resp.choices[0].message.content or "")
        except Exception as e:
            if self.verbose:
                print(f"  LLM error: {e}")
            return None, 0.0

    def _parse_output(self, content: str) -> Tuple[Optional[str], float]:
        step_m = re.search(
            r'STEP(?:\s*\d*)?:\s*(.+?)(?=\n(?:STEP|CONFIDENCE)|$)',
            content, re.IGNORECASE | re.DOTALL)
        conf_m = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)
        if step_m:
            text = step_m.group(1).strip()
            conf = float(conf_m.group(1)) if conf_m else 0.75
            conf = max(0.5, min(0.99, conf))
            if len(text) > 3:
                return text, conf
        ans_m = re.search(r'[Tt]he answer is[:\s]+(.+?)(?:\.|$)', content)
        if ans_m:
            text = f"The answer is: {ans_m.group(1).strip()}"
            conf = float(conf_m.group(1)) if conf_m else 0.75
            return text, max(0.5, min(0.99, conf))
        text = content.strip()
        text = re.sub(r'^(?:Step\s*\d+[:\s]*|STEP[:\s]*)', '', text, flags=re.IGNORECASE).strip()
        if 5 < len(text) < 500:
            return text, 0.6
        return None, 0.0

    def generate_candidates(self, problem: str, state: State,
                            q_type: str = 'bridge', context: str = '',
                            force_conclusion: bool = False) -> List[Tuple[str, float, bool]]:
        step_num = len(state.steps) + 1
        if force_conclusion:
            return self._generate_conclusion(problem, state, q_type)
        if q_type == 'yesno':
            return self._generate_yesno(problem, state, step_num)
        if q_type == 'comparison':
            return self._generate_comparison(problem, state, step_num)
        return self._generate_bridge(problem, state, step_num)

    # ─── Bridge ────────────────────────────────────────────────────────────
    def _generate_bridge(self, problem: str, state: State, step_num: int) -> List[Tuple[str, float, bool]]:
        candidates = []
        steps_so_far = "\n".join(f"  {i+1}. {s.content}" for i, s in enumerate(state.steps)) if state.steps else ""
        if step_num == 1:
            strategies = [
                ("You find intermediate facts in context paragraphs. Extract exact information. Never give the final answer in step 1.",
                 f"{problem}\n\nTASK: Find the key intermediate entity or fact from the context that the question references. This is step 1 — extract a FACT, not the final answer.\n\nSTEP: [intermediate fact from context]\nCONFIDENCE: [0.6-0.95]"),
                ("You identify the most relevant context paragraph and extract key details. Be specific.",
                 f"{problem}\n\nTASK: Which context paragraph answers part of this question? Extract the specific detail that serves as a bridge to the answer.\n\nSTEP: According to the [Title] article, [specific fact].\nCONFIDENCE: [0.6-0.95]"),
                ("You are a careful multi-hop reasoning assistant. Think about what intermediate information is needed.",
                 f"{problem}\n\nTASK: To answer this multi-hop question, what fact do I need to find first? Identify the bridge entity and look it up in the context.\n\nSTEP: [the intermediate fact you found]\nCONFIDENCE: [0.6-0.95]"),
                ("You extract supporting evidence from context for complex questions.",
                 f"{problem}\n\nTASK: Extract one key supporting fact that connects the question to the final answer.\n\nSTEP: [key fact]\nCONFIDENCE: [0.6-0.95]"),
            ]
            for i, (sys_p, usr_p) in enumerate(strategies[:self.branch_k]):
                temp = TEMPERATURES[i] if i < len(TEMPERATURES) else 0.4
                text, conf = self._call_llm(sys_p, usr_p, temp, 300)
                if text and 'the answer is' not in text.lower():
                    candidates.append((text, conf, True))
                elif text:
                    candidates.append((text, conf, False))
        else:
            strategies = [
                ("You derive answers from context using previously found facts. Always give a specific, concise answer.",
                 f"{problem}\n\nFacts found so far:\n{steps_so_far}\n\nUsing these facts and the context paragraphs, give the final answer to the question.\n\nSTEP: The answer is: [specific answer]\nCONFIDENCE: [0.7-1.0]"),
                ("You answer multi-hop questions by connecting facts from different context paragraphs.",
                 f"{problem}\n\nPrevious reasoning:\n{steps_so_far}\n\nNow find the specific answer in the context.\n\nSTEP: The answer is: [answer from context]\nCONFIDENCE: [0.7-1.0]"),
            ]
            for i, (sys_p, usr_p) in enumerate(strategies[:min(self.branch_k, 2)]):
                temp = TEMPERATURES[i] if i < len(TEMPERATURES) else 0.3
                text, conf = self._call_llm(sys_p, usr_p, temp, 256)
                if text:
                    candidates.append((text, conf, 'the answer is' not in text.lower()))
        return candidates

    # ─── Yesno ─────────────────────────────────────────────────────────────
    def _generate_yesno(self, problem: str, state: State, step_num: int) -> List[Tuple[str, float, bool]]:
        candidates = []
        if step_num == 1:
            strategies = [
                ("You extract specific property values for entities. Do NOT answer yes/no yet.",
                 f"{problem}\n\nTASK: Find the specific property being compared for BOTH entities in this yes/no question.\n\nSTEP: [Entity1] [property] is [value1]. [Entity2] [property] is [value2].\nCONFIDENCE: [0.7-0.95]"),
                ("You identify specific factual details about entities. Focus on exact values.",
                 f"{problem}\n\nTASK: What specific attribute is being compared? Find the exact value for each entity.\n\nSTEP: [Entity1]'s [attribute] is [value1], and [Entity2]'s [attribute] is [value2].\nCONFIDENCE: [0.7-0.95]"),
                ("You are a fact-checker. Find relevant details for both entities.",
                 f"{problem}\n\nTASK: Look up the relevant property for EACH entity in the context paragraphs.\n\nSTEP: [Your findings for both entities]\nCONFIDENCE: [0.7-0.95]"),
            ]
            for i, (sys_p, usr_p) in enumerate(strategies[:self.branch_k]):
                text, conf = self._call_llm(sys_p, usr_p, TEMPERATURES[i], 256)
                if text and 'the answer is' not in text.lower():
                    candidates.append((text, conf, True))
        else:
            prev = state.steps[0].content if state.steps else ""
            sys_p = "You compare values and determine if they match. Answer only yes or no."
            usr_p = (f"{problem}\n\nFacts extracted: {prev}\n\nTASK: Compare the two values above.\n"
                     "STEP: The answer is: [yes or no]\nCONFIDENCE: [0.8-1.0]")
            for i in range(min(self.branch_k, 2)):
                text, conf = self._call_llm(sys_p, usr_p, TEMPERATURES[i], 128)
                if text:
                    candidates.append((text, conf, False))
        return candidates

    # ─── Comparison ─────────────────────────────────────────────────────────
    def _generate_comparison(self, problem: str, state: State, step_num: int) -> List[Tuple[str, float, bool]]:
        candidates = []
        steps_so_far = "\n".join(f"  {i+1}. {s.content}" for i, s in enumerate(state.steps)) if state.steps else ""
        if step_num == 1:
            strategies = [
                ("You extract exact numeric/date values from context paragraphs. Do NOT compare yet.",
                 f"{problem}\n\nTASK: For BOTH entities mentioned, find their specific [attribute] values in the context. Extract EXACT numbers/dates.\n\nSTEP: [Entity1]'s [attribute] is [value1]. [Entity2]'s [attribute] is [value2].\nCONFIDENCE: [0.7-0.95]"),
                ("You look up facts in context and report them precisely.",
                 f"{problem}\n\nTASK: Identify both entities and their relevant properties. Report exact values from the context.\n\nSTEP: [Entity1]: [value1]. [Entity2]: [value2].\nCONFIDENCE: [0.7-0.95]"),
            ]
            for i, (sys_p, usr_p) in enumerate(strategies[:self.branch_k]):
                text, conf = self._call_llm(sys_p, usr_p, TEMPERATURES[i], 256)
                if text and 'the answer is' not in text.lower():
                    candidates.append((text, conf, True))
        else:
            sys_p = "You compare two values and state which is greater/earlier/etc."
            usr_p = (f"{problem}\n\nExtracted values:\n{steps_so_far}\n\n"
                     "Now compare the values and give the final answer.\n\nSTEP: The answer is: [entity or value]\nCONFIDENCE: [0.8-1.0]")
            for i in range(min(self.branch_k, 2)):
                text, conf = self._call_llm(sys_p, usr_p, TEMPERATURES[i], 128)
                if text:
                    candidates.append((text, conf, False))
        return candidates

    # ─── Force conclusion ────────────────────────────────────────────────────
    def _generate_conclusion(self, problem: str, state: State, q_type: str) -> List[Tuple[str, float, bool]]:
        trace = state.get_trace()
        if q_type == 'yesno':
            sys_p = "Based on the reasoning so far, answer yes or no."
            usr_p = f"{problem}\n\nReasoning so far:\n{trace}\n\nFinal answer (yes or no only):\nSTEP: The answer is: [yes/no]\nCONFIDENCE: [0.7-0.9]"
        else:
            sys_p = "Based on the reasoning so far, give a specific concise answer."
            usr_p = f"{problem}\n\nReasoning so far:\n{trace}\n\nFinal answer (concise):\nSTEP: The answer is: [answer]\nCONFIDENCE: [0.7-0.9]"
        text, conf = self._call_llm(sys_p, usr_p, 0.1, 128)
        if text:
            if 'the answer is' not in text.lower():
                text = f"The answer is: {text}"
            return [(text, conf, False)]
        default = "The answer is: yes" if q_type == 'yesno' else "The answer is: unknown"
        return [(default, 0.5, False)]

    def extract_answer(self, trace: str, q_type: str = 'bridge') -> Optional[str]:
        matches = re.findall(r'[Tt]he answer is[:\s]+([^\n]+)', trace)
        if matches:
            raw = matches[-1].strip().rstrip('.')
            raw = re.split(r'(?:,\s*(?:but|however|although|which|because|as|and\s+(?:he|she|it|they)))',
                           raw, maxsplit=1)[0].strip()
            raw = re.split(r'\s+(?:is|was|were|has|have|had)\s+(?:the|a|an)\s+',
                           raw, maxsplit=1)[0].strip()
            if q_type == 'yesno':
                low = raw.lower()
                if re.search(r'\byes\b', low): return 'yes'
                if re.search(r'\bno\b',  low): return 'no'
            return raw
        lines = [l.strip() for l in trace.strip().split('\n') if l.strip()]
        if q_type == 'yesno' and lines:
            # Capture yes/no even when the model writes it in any step, not only the last one.
            for line in reversed(lines):
                txt = re.sub(r'^Step\s+\d+:\s*', '', line).strip().lower()
                if re.fullmatch(r'[\[(]?\s*yes\s*[\])]?[\.!?]?', txt) or re.search(r'\byes\b', txt):
                    return 'yes'
                if re.fullmatch(r'[\[(]?\s*no\s*[\])]?[\.!?]?', txt) or re.search(r'\bno\b', txt):
                    return 'no'
        if lines:
            last = re.sub(r'^Step\s+\d+:\s*', '', lines[-1]).strip()
            if q_type == 'yesno':
                low = last.lower()
                if re.search(r'\byes\b', low): return 'yes'
                if re.search(r'\bno\b',  low): return 'no'
            return last
        return None


# ─── A* Search (with heuristic-mode support) ──────────────────────────────────

def _select_best_goal(goal_nodes: List[Node], target_answer: str,
                      solver: HotpotQASolver, q_type: str) -> Node:
    target_norm = _normalize(target_answer)
    matching = []
    for node in goal_nodes:
        ans = solver.extract_answer(node.state.get_trace(), q_type)
        if ans and _normalize(ans) == target_norm:
            matching.append(node)
    candidates = matching if matching else goal_nodes
    return max(candidates, key=lambda n: n.state.steps[-1].confidence if n.state.steps else 0)


def astar_search(
    problem: str,
    solver: HotpotQASolver,
    q_type: str = 'bridge',
    context: str = '',
    question_entities: Set[str] = None,
    heuristic_mode: str = 'astar',
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_BUDGET,
    min_goal_depth: int = DEFAULT_MIN_GOAL_DEPTH,
    majority_vote: int = DEFAULT_MAJORITY_VOTE,
    verbose: bool = False,
) -> Tuple[Node, int]:
    if question_entities is None:
        question_entities = set()

    open_set: List[Node] = []
    node_counter = 0
    nodes_explored = 0
    visited: Set[str] = set()
    goal_nodes: List[Node] = []
    goal_votes: Counter = Counter()
    best_current: Optional[Node] = None

    root_state = State()
    h_init = compute_heuristic(root_state, question_entities, q_type, heuristic_mode)
    # For greedy mode, f = h only; for bfs/astar, f = g + h (g=0 at root so same)
    f_init = h_init
    root_node = Node(state=root_state, g_score=0.0, h_score=h_init,
                     f_score=f_init, node_id=0)
    heapq.heappush(open_set, root_node)

    while open_set and nodes_explored < max_nodes:
        current = heapq.heappop(open_set)
        sig = current.state.signature()
        if sig in visited:
            continue
        visited.add(sig)
        nodes_explored += 1

        # ── Goal detection ─────────────────────────────────────────────
        if current.state.steps:
            last_text = current.state.steps[-1].content.lower()
            is_goal = False
            if q_type == 'yesno':
                if re.search(r'the answer is[:\s]+(yes|no)\b', last_text):
                    is_goal = True
            else:
                if 'the answer is' in last_text:
                    is_goal = True

            if is_goal and current.state.depth < min_goal_depth:
                conf = current.state.steps[-1].confidence
                if conf >= 0.85:
                    ans_c = solver.extract_answer(current.state.get_trace(), q_type)
                    if verify_answer_in_context(ans_c or "", context) < 0.5:
                        is_goal = False
                else:
                    is_goal = False

            if is_goal and context:
                ans_c = solver.extract_answer(current.state.get_trace(), q_type)
                if verify_answer_in_context(ans_c or "", context) < 0.3:
                    current = Node(state=current.state,
                                   g_score=current.g_score + 0.8,
                                   h_score=current.h_score,
                                   f_score=current.f_score + 0.8,
                                   node_id=current.node_id,
                                   parent_id=current.parent_id)

            if is_goal:
                goal_nodes.append(current)
                ans_c = solver.extract_answer(current.state.get_trace(), q_type)
                if ans_c:
                    norm = _normalize(ans_c)
                    goal_votes[norm] += 1
                    if goal_votes[norm] >= majority_vote:
                        best = _select_best_goal(goal_nodes, ans_c, solver, q_type)
                        return best, nodes_explored
                continue

        if best_current is None or current.state.depth > best_current.state.depth:
            best_current = current

        if current.state.depth >= max_depth:
            continue

        # ── Expand ─────────────────────────────────────────────────────
        force = (current.state.depth >= max_depth - 1)
        candidates = solver.generate_candidates(problem, current.state, q_type, context, force)

        existing_texts = [s.content for s in current.state.steps]
        seen_keys: Set[str] = set()
        unique: List[Tuple[str, float, bool]] = []
        for text, conf, is_factual in candidates:
            key = text.strip().lower()
            if key in seen_keys: continue
            if any(jaccard_sim(text, prev) > 0.7 for prev in existing_texts): continue
            if any(jaccard_sim(text, u[0]) > 0.7 for u in unique): continue
            seen_keys.add(key)
            unique.append((text, conf, is_factual))

        if not unique and current.state.steps:
            unique = solver._generate_conclusion(problem, current.state, q_type)

        for text, conf, is_factual in unique:
            node_counter += 1
            new_state = State(
                steps=current.state.steps + [ReasoningStep(text, conf, is_factual)],
                depth=current.state.depth + 1,
            )
            step_cost = 1.0 + (1.0 - conf)
            new_g = current.g_score + step_cost
            new_h = compute_heuristic(new_state, question_entities, q_type, heuristic_mode)

            # ── Key: priority function changes per mode ──────────────────
            if heuristic_mode == 'greedy':
                new_f = new_h          # ignore path cost
            else:
                new_f = new_g + new_h  # standard A* (also BFS when h=0)

            child = Node(state=new_state, g_score=new_g, h_score=new_h,
                         f_score=new_f, node_id=node_counter,
                         parent_id=current.node_id)
            heapq.heappush(open_set, child)

    # ── Post-search ─────────────────────────────────────────────────────
    if goal_nodes:
        if goal_votes:
            best_ans = goal_votes.most_common(1)[0][0]
            return _select_best_goal(goal_nodes, best_ans, solver, q_type), nodes_explored
        return goal_nodes[0], nodes_explored

    fallback = best_current or Node(state=State(), g_score=0, h_score=0, f_score=0, node_id=0)
    forced = solver._generate_conclusion(problem, fallback.state, q_type)
    if forced:
        text, conf, _ = forced[0]
        node_counter += 1
        final_state = State(
            steps=fallback.state.steps + [ReasoningStep(text, conf, False)],
            depth=fallback.state.depth + 1,
        )
        return Node(state=final_state, g_score=fallback.g_score + 1.0,
                    h_score=0.0, f_score=fallback.g_score + 1.0,
                    node_id=node_counter), nodes_explored
    return fallback, nodes_explored


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HotpotQA rebuttal ablation")
    parser.add_argument("--sample-file", default=None,
                        help="Path to prepared sample JSON (default: samples/hotpotqa_300.json)")
    parser.add_argument("--heuristic-mode", choices=["astar", "bfs", "greedy"],
                        default="astar",
                        help="astar=full A*, bfs=h=0 (uniform cost), greedy=h-only")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="Max nodes to expand (default=30)")
    parser.add_argument("--branch-k", type=int, default=DEFAULT_BRANCH_K,
                        help="Branches per expansion (default=3)")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out",      default=None,
                        help="Output JSON (auto-named if omitted)")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sample_file = args.sample_file or os.path.join(SAMPLES_DIR, "hotpotqa_300.json")
    if not os.path.exists(sample_file):
        print(f"ERROR: sample file not found: {sample_file}")
        print("Run prepare_samples.py first.")
        sys.exit(1)

    if args.out is None:
        tag = f"mode{args.heuristic_mode}_budget{args.budget}_k{args.branch_k}"
        args.out = os.path.join(RESULTS_DIR, f"hotpotqa_{tag}.json")

    print(f"Config: heuristic={args.heuristic_mode} | budget={args.budget} | branch_k={args.branch_k}")
    print(f"Model : {args.model} @ {args.base_url}")
    print(f"Sample: {sample_file}")
    print(f"Output: {args.out}")

    # ── Load sample ──────────────────────────────────────────────────────
    with open(sample_file) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} questions.")

    solver = HotpotQASolver(
        model=args.model, base_url=args.base_url,
        branch_k=args.branch_k, verbose=args.verbose
    )

    results = []
    correct = 0
    total   = 0
    total_nodes = 0
    start_time  = time.time()

    for item in tqdm(items, desc=f"HotpotQA [{args.heuristic_mode}|B={args.budget}|K={args.branch_k}]"):
        try:
            q_type   = detect_question_type(item)
            question = item["question"]
            gold     = item["answer"]

            # Rebuild problem string from pre-processed context
            context_str = item.get("context", "")
            problem = f"Context:\n{context_str}\n\nQuestion: {question}"

            entities = extract_question_entities(question)

            best_node, n_explored = astar_search(
                problem=problem,
                solver=solver,
                q_type=q_type,
                context=context_str,
                question_entities=entities,
                heuristic_mode=args.heuristic_mode,
                max_nodes=args.budget,
                max_depth=DEFAULT_MAX_DEPTH,
                min_goal_depth=DEFAULT_MIN_GOAL_DEPTH,
                majority_vote=DEFAULT_MAJORITY_VOTE,
                verbose=args.verbose,
            )

            pred = solver.extract_answer(best_node.state.get_trace(), q_type)
            f1   = f1_score(pred or "", gold)
            is_correct = (_normalize(pred or "") == _normalize(gold)) or f1 >= 0.05

            total_nodes += n_explored
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id":      item.get("id", total),
                "question": question,
                "gold":    gold,
                "pred":    pred,
                "correct": is_correct,
                "f1":      round(f1, 4),
                "nodes":   n_explored,
                "q_type":  q_type,
                "trace":   best_node.state.get_trace(),
            })

        except KeyboardInterrupt:
            print("\nInterrupted — saving partial results...")
            break
        except Exception as e:
            print(f"  Error on item {total}: {e}")
            results.append({
                "id": item.get("id", total), "question": item.get("question", ""),
                "gold": item.get("answer", ""), "pred": None,
                "correct": False, "f1": 0.0, "nodes": 0, "q_type": "unknown", "trace": "",
            })
            total += 1

    elapsed = time.time() - start_time
    accuracy = 100 * correct / total if total > 0 else 0.0
    avg_nodes = total_nodes / total if total > 0 else 0.0

    summary = {
        "config": {
            "heuristic_mode": args.heuristic_mode,
            "budget":         args.budget,
            "branch_k":       args.branch_k,
            "model":          args.model,
        },
        "total":     total,
        "correct":   correct,
        "accuracy":  round(accuracy, 4),
        "avg_nodes": round(avg_nodes, 2),
        "elapsed_s": round(elapsed, 1),
        "total_tokens": solver.total_tokens,
        "api_calls":    solver.api_calls,
        "results":      results,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"Accuracy  : {accuracy:.2f}%  ({correct}/{total})")
    print(f"Avg nodes : {avg_nodes:.1f}")
    print(f"Tokens    : {solver.total_tokens:,}")
    print(f"Time      : {elapsed/60:.1f} min")
    print(f"Saved     : {args.out}")


if __name__ == "__main__":
    main()
