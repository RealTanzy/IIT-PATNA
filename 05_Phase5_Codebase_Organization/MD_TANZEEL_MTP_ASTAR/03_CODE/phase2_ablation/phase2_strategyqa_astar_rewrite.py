#!/usr/bin/env python3
"""
Phase 2 StrategyQA A* rewrite with explicit meta-prompting.

What is new compared to previous solver variants:
1) Prompt logic is externalized in PHASE_2/meta_prompts.py
2) Node records keep prompt version for auditability
3) Search attempts branch diversification and refinement steps
4) JSON output includes trace metadata needed for Phase 2 analysis
"""

import argparse
import heapq
import json
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from openai import OpenAI
from tqdm import tqdm

from meta_prompts import (
    PROMPT_VERSION,
    conclude_yes_no,
    thought_decompose,
    thought_entity_lookup,
    thought_refine,
)


MODEL = os.getenv("PHASE2_MODEL", "qwen3-coder-next:latest")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

DEFAULT_LIMIT = 200

BRANCH_K = 3
MAX_DEPTH = 8
MAX_NODES = 20
MIN_GOAL_DEPTH = 2
MAJORITY_VOTE = 2
TEMPERATURES = [0.2, 0.45, 0.7]

W_DEPTH = 0.3
W_COVERAGE = 0.3
HEURISTIC_MODE = "full"

STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "were", "are", "of", "in", "on", "at",
    "for", "and", "or", "to", "by", "with", "it", "its", "that", "this",
    "from", "as", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can", "what",
    "which", "who", "where", "when", "how", "why", "same", "both", "also",
    "not", "no", "yes", "than", "more", "most", "other", "some", "any", "all",
    "many", "much", "few", "several", "first", "last", "new", "old",
})


@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.75
    prompt_name: str = ""


@dataclass
class State:
    steps: List[ReasoningStep] = field(default_factory=list)
    depth: int = 0

    def trace(self) -> str:
        return "\n".join(f"Step {i+1}: {step.content}" for i, step in enumerate(self.steps))

    def signature(self) -> str:
        return " || ".join(s.content.strip().lower() for s in self.steps)


@dataclass
class Node:
    state: State
    g: float
    h: float
    f: float
    node_id: int
    parent_id: Optional[int] = None

    def __lt__(self, other: "Node") -> bool:
        if abs(self.f - other.f) > 1e-9:
            return self.f < other.f
        return self.node_id < other.node_id


def extract_entities(question: str) -> Set[str]:
    entities: Set[str] = set()

    for m in re.finditer(r"\"([^\"]+)\"", question):
        entities.add(m.group(1).lower())

    q2 = re.sub(
        r"^(What|Which|Who|Where|When|How|Are|Were|Is|Was|Did|Do|Does|Could|Would|Can|Has|Have|Had|The)\s+",
        "",
        question,
        flags=re.IGNORECASE,
    )
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", q2):
        phrase = m.group(1).lower()
        if phrase not in STOPWORDS and len(phrase) > 2:
            entities.add(phrase)

    for token in re.findall(r"\b[a-zA-Z]+\b", question.lower()):
        if token not in STOPWORDS and len(token) > 4:
            entities.add(token)

    return entities


def entity_coverage(state: State, question_entities: Set[str]) -> float:
    if not question_entities:
        return 1.0
    trace_lower = state.trace().lower()
    covered = sum(1 for ent in question_entities if ent in trace_lower)
    return covered / len(question_entities)


def heuristic(state: State, question_entities: Set[str]) -> float:
    if HEURISTIC_MODE == "none":
        return 0.0
    if HEURISTIC_MODE == "depth":
        return max(0, 2 - state.depth) * W_DEPTH
    if HEURISTIC_MODE == "coverage":
        return (1.0 - entity_coverage(state, question_entities)) * W_COVERAGE

    h_depth = max(0, 2 - state.depth) * W_DEPTH
    h_cov = (1.0 - entity_coverage(state, question_entities)) * W_COVERAGE
    return h_depth + h_cov


def jaccard_similarity(a: str, b: str) -> float:
    aw = set(a.lower().split())
    bw = set(b.lower().split())
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


class Phase2StrategyQASolver:
    def __init__(self, model: str, verbose: bool = False):
        self.client = OpenAI(base_url=BASE_URL, api_key="ollama")
        self.model = model
        self.verbose = verbose
        self.total_tokens = 0
        self.api_calls = 0
        self.model_fallbacks = [
            model,
            "qwen3-coder-next:latest",
            "llama3.2:3b",
            "llama3.1:8b",
            "qwen2.5:0.5b",
        ]

    def _call_llm(self, system_prompt: str, user_prompt: str, temp: float, max_tokens: int) -> Tuple[Optional[str], float]:
        last_error = None
        for candidate in self.model_fallbacks:
            try:
                resp = self.client.chat.completions.create(
                    model=candidate,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                    max_tokens=max_tokens,
                    stop=["\n\n\n"],
                )
                self.api_calls += 1
                if resp.usage:
                    self.total_tokens += resp.usage.total_tokens
                if candidate != self.model and self.verbose:
                    print(f"Model fallback in use: {candidate}")
                self.model = candidate
                content = (resp.choices[0].message.content or "").strip()
                return self._parse_output(content)
            except Exception as exc:
                last_error = exc
                continue

        if self.verbose and last_error is not None:
            print(f"LLM call failed after fallbacks: {last_error}")
        return None, 0.0

    @staticmethod
    def _parse_output(content: str) -> Tuple[Optional[str], float]:
        if not content:
            return None, 0.0

        step_match = re.search(
            r"STEP(?:\s*\d*)?:\s*(.+?)(?=\n(?:STEP|CONFIDENCE)|$)",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", content, flags=re.IGNORECASE)

        if step_match:
            text = step_match.group(1).strip()
            conf = float(conf_match.group(1)) if conf_match else 0.75
            return text, max(0.5, min(0.99, conf))

        # Fall back to free text if model ignored formatting.
        text = content.strip()
        if 3 < len(text) < 600:
            conf = float(conf_match.group(1)) if conf_match else 0.6
            return text, max(0.5, min(0.99, conf))

        return None, 0.0

    def extract_answer(self, trace: str) -> Optional[str]:
        matches = re.findall(r"[Tt]he answer is[:\s]+([^\n]+)", trace)
        if matches:
            last = matches[-1].strip().lower().rstrip(".")
            if re.search(r"\byes\b", last):
                return "yes"
            if re.search(r"\bno\b", last):
                return "no"

        lines = [ln.strip().lower() for ln in trace.splitlines() if ln.strip()]
        if lines:
            last_line = lines[-1]
            if re.search(r"\byes\b", last_line):
                return "yes"
            if re.search(r"\bno\b", last_line):
                return "no"
        return None

    def generate_candidates(self, question: str, state: State, force_conclusion: bool = False) -> List[Tuple[ReasoningStep, float]]:
        candidates: List[Tuple[ReasoningStep, float]] = []
        trace = state.trace()

        if force_conclusion:
            prompt = conclude_yes_no(question, trace)
            text, conf = self._call_llm(prompt.system, prompt.user, temp=0.1, max_tokens=128)
            if text:
                if "the answer is" not in text.lower():
                    text = f"The answer is: {text}"
                step = ReasoningStep(content=text, confidence=conf, prompt_name="conclude_yes_no")
                candidates.append((step, conf))
            return candidates

        step_num = state.depth + 1

        if step_num == 1:
            specs = [
                ("thought_decompose", thought_decompose(question), TEMPERATURES[0], 300),
                ("thought_entity_lookup", thought_entity_lookup(question), TEMPERATURES[1], 300),
            ]
        else:
            specs = [
                ("thought_refine", thought_refine(question, trace), TEMPERATURES[0], 220),
                ("conclude_yes_no", conclude_yes_no(question, trace), TEMPERATURES[1], 128),
                ("conclude_yes_no", conclude_yes_no(question, trace), TEMPERATURES[2], 128),
            ]

        for prompt_name, spec, temp, max_tok in specs[:BRANCH_K]:
            text, conf = self._call_llm(spec.system, spec.user, temp=temp, max_tokens=max_tok)
            if not text:
                continue
            step = ReasoningStep(content=text, confidence=conf, prompt_name=prompt_name)
            candidates.append((step, conf))

        return candidates


def select_best_goal(goals: List[Node], target: str, solver: Phase2StrategyQASolver) -> Node:
    matching = [
        node
        for node in goals
        if (solver.extract_answer(node.state.trace()) or "") == target
    ]
    if matching:
        return max(matching, key=lambda node: node.state.steps[-1].confidence)
    return min(goals, key=lambda node: node.f)


def run_astar(question: str, solver: Phase2StrategyQASolver) -> Tuple[Node, int]:
    entities = extract_entities(question)

    root = Node(state=State(), g=0.0, h=0.0, f=0.0, node_id=0)
    root.h = heuristic(root.state, entities)
    root.f = root.h

    open_set: List[Node] = [root]
    visited: Set[str] = set()
    goals: List[Node] = []
    votes: Counter = Counter()

    node_counter = 0
    explored = 0
    fallback = root

    # Phase 2 rewrite starts with one full reasoning seed to improve early direction.
    cot_prompt = conclude_yes_no(question, "(none yet)")
    seed_text, seed_conf = solver._call_llm(cot_prompt.system, cot_prompt.user, temp=0.2, max_tokens=256)
    if seed_text:
        if "the answer is" not in seed_text.lower():
            seed_text = f"The answer is: {seed_text}"
        node_counter += 1
        seed_state = State(
            steps=[ReasoningStep(seed_text, seed_conf, "seed_conclusion")],
            depth=2,
        )
        seed_g = 1.0 + (1.0 - seed_conf)
        seed_h = heuristic(seed_state, entities)
        heapq.heappush(
            open_set,
            Node(seed_state, seed_g, seed_h, seed_g + seed_h, node_counter, 0),
        )

    while open_set and explored < MAX_NODES:
        current = heapq.heappop(open_set)
        signature = current.state.signature()
        if signature in visited:
            continue
        visited.add(signature)
        explored += 1

        if current.state.depth > fallback.state.depth:
            fallback = current

        if current.state.steps:
            answer = solver.extract_answer(current.state.trace())
            if answer and current.state.depth >= MIN_GOAL_DEPTH:
                goals.append(current)
                votes[answer] += 1
                if votes[answer] >= MAJORITY_VOTE:
                    return select_best_goal(goals, answer, solver), explored
                continue

        if current.state.depth >= MAX_DEPTH:
            continue

        force_conclusion = current.state.depth >= MAX_DEPTH - 1
        raw_candidates = solver.generate_candidates(
            question,
            current.state,
            force_conclusion=force_conclusion,
        )

        if not raw_candidates and current.state.steps:
            raw_candidates = solver.generate_candidates(question, current.state, force_conclusion=True)

        existing = [s.content for s in current.state.steps]
        deduped: List[Tuple[ReasoningStep, float]] = []
        seen: Set[str] = set()

        for step, conf in raw_candidates:
            key = step.content.strip().lower()
            if key in seen:
                continue
            if any(jaccard_similarity(step.content, prev) > 0.72 for prev in existing):
                continue
            if any(jaccard_similarity(step.content, d[0].content) > 0.72 for d in deduped):
                continue
            seen.add(key)
            deduped.append((step, conf))

        for step, conf in deduped:
            node_counter += 1
            new_state = State(
                steps=current.state.steps + [step],
                depth=current.state.depth + 1,
            )
            step_cost = 1.0 + (1.0 - conf)
            g = current.g + step_cost
            h = heuristic(new_state, entities)
            child = Node(new_state, g, h, g + h, node_counter, current.node_id)
            heapq.heappush(open_set, child)

    if goals:
        answer, count = votes.most_common(1)[0]
        if count >= 1:
            return select_best_goal(goals, answer, solver), explored
        return min(goals, key=lambda node: node.f), explored

    # Force one final conclusion from the best fallback branch.
    forced = solver.generate_candidates(question, fallback.state, force_conclusion=True)
    if forced:
        step, _ = forced[0]
        final_state = State(
            steps=fallback.state.steps + [step],
            depth=fallback.state.depth + 1,
        )
        node_counter += 1
        final_node = Node(
            state=final_state,
            g=fallback.g + 1.0,
            h=0.0,
            f=fallback.g + 1.0,
            node_id=node_counter,
            parent_id=fallback.node_id,
        )
        return final_node, explored

    return fallback, explored


def normalize_gold(answer) -> str:
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    text = str(answer).strip().lower()
    if text in {"true", "1"}:
        return "yes"
    if text in {"false", "0"}:
        return "no"
    return text


def load_strategyqa(path: str) -> List[dict]:
    with open(path) as f:
        raw = json.load(f)

    items: List[dict] = []
    for row in raw:
        if "question" not in row:
            continue
        gold = row.get("gold", row.get("answer", row.get("label")))
        if gold is None:
            continue
        items.append({"question": row["question"], "answer": normalize_gold(gold)})
    return items


def build_result(question: str, gold: str, pred: Optional[str], node: Node, nodes_explored: int) -> dict:
    correct = pred == gold if pred is not None else False

    step_rows = []
    for idx, step in enumerate(node.state.steps, start=1):
        step_rows.append(
            {
                "step_number": idx,
                "content": step.content,
                "confidence": step.confidence,
                "prompt_name": step.prompt_name,
                "prompt_version": PROMPT_VERSION,
            }
        )

    return {
        "question": question,
        "gold": gold,
        "pred": pred,
        "correct": correct,
        "nodes": nodes_explored,
        "trace": node.state.trace(),
        "prompt_version": PROMPT_VERSION,
        "steps": step_rows,
    }


def main() -> None:
    global BRANCH_K, MAX_DEPTH, MAX_NODES, MIN_GOAL_DEPTH
    global MAJORITY_VOTE, TEMPERATURES, W_DEPTH, W_COVERAGE, HEURISTIC_MODE
    global BASE_URL

    parser = argparse.ArgumentParser(description="Phase 2 StrategyQA A* rewrite with meta-prompting")
    parser.add_argument("--data-path", type=str, required=True, help="Path to local StrategyQA JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--base-url", type=str, default=BASE_URL,
                        help="Ollama OpenAI-compatible base URL")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--branch-k", type=int, default=BRANCH_K)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES)
    parser.add_argument("--min-goal-depth", type=int, default=MIN_GOAL_DEPTH)
    parser.add_argument("--majority-vote", type=int, default=MAJORITY_VOTE)
    parser.add_argument("--temps", type=str, default=",".join(str(t) for t in TEMPERATURES),
                        help="Comma-separated temperatures, e.g. 0.2,0.45,0.7")
    parser.add_argument("--w-depth", type=float, default=W_DEPTH)
    parser.add_argument("--w-coverage", type=float, default=W_COVERAGE)
    parser.add_argument("--heuristic-mode", choices=["full", "depth", "coverage", "none"],
                        default=HEURISTIC_MODE)
    args = parser.parse_args()

    BASE_URL = args.base_url

    BRANCH_K = max(1, args.branch_k)
    MAX_DEPTH = max(1, args.max_depth)
    MAX_NODES = max(1, args.max_nodes)
    MIN_GOAL_DEPTH = max(1, args.min_goal_depth)
    MAJORITY_VOTE = max(1, args.majority_vote)
    W_DEPTH = max(0.0, args.w_depth)
    W_COVERAGE = max(0.0, args.w_coverage)
    HEURISTIC_MODE = args.heuristic_mode

    parsed_temps = []
    for x in args.temps.split(","):
        x = x.strip()
        if not x:
            continue
        parsed_temps.append(float(x))
    if parsed_temps:
        TEMPERATURES = parsed_temps

    items = load_strategyqa(args.data_path)
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(items)
    items = items[: args.limit]

    solver = Phase2StrategyQASolver(model=args.model, verbose=args.verbose)

    out_path = args.output
    if not out_path:
        os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
        model_tag = args.model.replace(":", "_").replace("/", "_")
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{model_tag}_phase2_prompted_astar_{args.seed}.json",
        )

    results: List[dict] = []
    correct = 0
    total = 0
    started = time.time()

    for item in tqdm(items, desc="Phase2 A* rewrite"):
        question = item["question"]
        gold = item["answer"]

        try:
            best, explored = run_astar(question, solver)
            pred = solver.extract_answer(best.state.trace())
            result = build_result(question, gold, pred, best, explored)
            results.append(result)
            total += 1
            if result["correct"]:
                correct += 1
        except Exception as exc:
            if args.verbose:
                print(f"Question failed: {exc}")
            results.append(
                {
                    "question": question,
                    "gold": gold,
                    "pred": None,
                    "correct": False,
                    "nodes": 0,
                    "trace": "",
                    "prompt_version": PROMPT_VERSION,
                    "steps": [],
                    "error": str(exc),
                }
            )
            total += 1

        # Crash-safe write after each sample.
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    elapsed = time.time() - started
    accuracy = (correct / total * 100.0) if total else 0.0
    avg_nodes = sum(r.get("nodes", 0) for r in results) / total if total else 0.0

    print("=" * 72)
    print("Phase 2 A* Rewrite + Meta-Prompting")
    print("=" * 72)
    print(f"Model:        {args.model}")
    print(f"Base URL:     {BASE_URL}")
    print(f"Heuristic:    {HEURISTIC_MODE} (w_depth={W_DEPTH}, w_coverage={W_COVERAGE})")
    print(f"Search cfg:   branch={BRANCH_K}, max_nodes={MAX_NODES}, max_depth={MAX_DEPTH}")
    print(f"Examples:     {total}")
    print(f"Accuracy:     {correct}/{total} ({accuracy:.1f}%)")
    print(f"Avg nodes:    {avg_nodes:.2f}")
    print(f"API calls:    {solver.api_calls}")
    print(f"Total tokens: {solver.total_tokens}")
    print(f"Prompt ver:   {PROMPT_VERSION}")
    print(f"Elapsed:      {elapsed:.1f}s")
    print(f"Saved:        {out_path}")


if __name__ == "__main__":
    main()
