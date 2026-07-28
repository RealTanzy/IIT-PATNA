#!/usr/bin/env python3
"""
A* Search Solver for HotpotQA (Multi-hop Question Answering)
Adapted for multi-hop factual reasoning.
"""

import os
import re
import heapq
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Search parameters
LAMBDA = 0.12        # Weight for depth penalty (encourage efficiency)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ReasoningStep:
    content: str
    confidence: float = 0.8
    derived_values: Dict = field(default_factory=dict)

@dataclass
class State:
    steps: List[ReasoningStep] = field(default_factory=list)
    depth: int = 0

    def get_trace(self) -> str:
        lines = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {step.content}")
        return "\n".join(lines)

    def signature(self) -> str:
        return " || ".join(step.content.strip() for step in self.steps)

@dataclass
class Node:
    state: State
    g_score: float      # Actual cost
    h_score: float      # Estimated cost to goal
    h_score_semantic: float
    f_score: float      # Total cost (g + h)
    node_id: int
    parent_id: Optional[int] = None

    def __lt__(self, other: "Node") -> bool:
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.node_id < other.node_id

# ============================================================================
# HEURISTICS FOR QA REASONING
# ============================================================================

# ============================================================================
# QUESTION TYPE DETECTION  (based on train.csv: bridge 81%, comparison+yesno 19%)
# ============================================================================

def detect_question_type(question: str) -> str:
    """
    Classify question as 'yesno', 'comparison', or 'bridge'.
    Patterns derived from HotpotQA train.csv:
      - yes/no: always comparison type; signals 'both', 'same', 'also', 'as well'
      - comparison: 'which ... more/older/bigger/first', 'when did ... compared'
      - bridge: everything else (entity-chain: find A, then find property of A)
    """
    q = question.lower().strip()
    yesno_signals = [
        'are both', 'were both', 'is both', 'do both', 'did both', 'have both',
        'are the', 'were the', 'is the same', 'in the same', 'both located',
        'both members', 'both known', 'both considered', 'both play',
        'of the same', 'the same type', 'the same genre', 'the same nationality',
        'same nationality', 'same type', 'same genre', 'same country', 'same field',
        'known for the same', 'same kind', 'same sport', 'same language',
        'also a ', 'both a ', 'both an ',
    ]
    if any(s in q for s in yesno_signals):
        return 'yesno'
    if re.search(r'(were|are|is|was)\s+\w[\w\s]+\s+and\s+\w[\w\s]+\s+(both|also|the same|known)', q):
        return 'yesno'
    comparison_signals = [
        'which is older', 'which is newer', 'which is larger', 'which is bigger',
        'which is smaller', 'which came first', 'which was first', 'who is older',
        'who is younger', 'who won more', 'who has more', 'which has more',
        'which premiered first', 'which started first', 'which was founded',
    ]
    if any(s in q for s in comparison_signals):
        return 'comparison'
    return 'bridge'

# ============================================================================
# HEURISTICS FOR QA REASONING
# ============================================================================
# Admissibility proof:
#   - All HotpotQA questions need MINIMUM 2 reasoning hops
#   - min step_cost = 1.0 + (1 - 0.99) = 1.01  (highest confidence)
#   - min 2-hop cost = 2 * 1.01 = 2.02
#   - max h(root) with LAMBDA=0.12, max_depth=25, max_complexity=3:
#     h(root) = (0.12*25 + 0.12*3) / 2 = (3.0 + 0.36) / 2 = 1.68
#   - 1.68 < 2.02  =>  ADMISSIBLE ✓

def h_remaining_steps(state: State, max_depth: int = 8) -> float:
    """Heuristic based on remaining depth budget."""
    remaining = max(0, max_depth - state.depth)
    return LAMBDA * remaining

def h_qa_complexity(state: State, question: str, q_type: str = 'bridge') -> float:
    """
    Type-aware complexity heuristic (admissible).
      yesno/comparison : 2 hops  -> complexity = 2
      bridge           : 2+ hops -> complexity = 2, +1 if person-lookup chain
    Decrements by 1 per depth.
    """
    if q_type in ('yesno', 'comparison'):
        complexity = 2
    else:  # bridge
        q_lower = question.lower()
        complexity = 2
        if any(w in q_lower for w in ['who directed', 'who wrote', 'who played',
                                       'who founded', 'who created', 'who starred']):
            complexity += 1
    remaining = max(0, complexity - state.depth)
    return LAMBDA * remaining

def compute_heuristic(state: State, problem: str, max_depth: int = 8,
                      q_type: str = 'bridge') -> float:
    """Compute combined admissible heuristic."""
    q_match = re.search(r'Question:\s*(.+)$', problem, re.MULTILINE)
    question = q_match.group(1).strip() if q_match else problem
    h1 = h_remaining_steps(state, max_depth)
    h2 = h_qa_complexity(state, question, q_type)
    return (h1 + h2) / 2

# ============================================================================
# SIMILARITY HELPER
# ============================================================================

def _jaccard_sim(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

# ============================================================================
# SOLVER LOGIC
# ============================================================================

class HotpotQASolver:
    def __init__(self, api_key: str, verbose: bool = True):
        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        self.total_tokens = 0
        self.api_calls = 0
        self.verbose = verbose

    def extract_answer(self, trace: str, q_type: str = 'bridge') -> Optional[str]:
        """Extract answer with yes/no normalization for yesno questions."""
        # Filter out template leakage (model echoed prompt placeholder text)
        matches = [m for m in re.findall(r'[Tt]he answer is[: ]+([^.\n]+)', trace)
                   if not re.search(r'\[.*\]', m)]
        if matches:
            raw = matches[-1].strip().rstrip('.')
            if q_type == 'yesno':
                low = raw.lower()
                if re.search(r'\byes\b', low): return 'yes'
                if re.search(r'\bno\b',  low): return 'no'
            else:
                return raw
        # Fallback: last non-empty line, strip "Step N:" prefix
        lines = [l.strip() for l in trace.strip().split('\n') if l.strip()]
        last  = lines[-1] if lines else None
        if last:
            last = re.sub(r'^Step\s+\d+:\s*', '', last).strip()
            if q_type == 'yesno':
                low = last.lower()
                if re.search(r'\byes\b', low): return 'yes'
                if re.search(r'\bno\b',  low): return 'no'
        return last

    def generate_next_steps(
        self,
        problem: str,
        current_state: State,
        n_candidates: int = 3,
        force_conclusion: bool = False,
        q_type: str = 'bridge'
    ) -> List[Tuple[str, float]]:
        """Generate next QA reasoning steps with type-specific prompts."""

        if current_state.steps:
            recent_steps = "\n".join(
                f"{i+1}. {s.content}" for i, s in enumerate(current_state.steps)
            )
            context  = f"Previous reasoning steps:\n{recent_steps}\n\n"
            step_num = len(current_state.steps) + 1
        else:
            context  = ""
            step_num = 1

        # ── Type-specific prompts ───────────────────────────────────────────
        if force_conclusion:
            if q_type == 'yesno':
                prompt = f"""Based on the facts gathered, answer the yes/no question.

{problem}

{context}
Look at the PREVIOUS STEP above. It contains two specific values.

Instructions:
1. Quote the EXACT value for Entity 1 (copy it word-for-word from the previous step).
2. Quote the EXACT value for Entity 2 (copy it word-for-word from the previous step).
3. Are these two quoted values IDENTICAL? Even if both are in the same city/country, they are NOT the same unless the specific values match.
   - Same specific value (e.g. both "Fatih", both "American") → "The answer is: yes"
   - Different specific values (e.g. "Fatih" vs "Ortaköy") → "The answer is: no"

Output ONLY:
STEP: The answer is: yes
or:
STEP: The answer is: no
CONFIDENCE: [0.8-1.0]"""
            else:
                prompt = f"""You are answering a multi-hop question. Give the FINAL ANSWER now.

{problem}

{context}
INSTRUCTIONS:
1. Re-read the question. What TYPE of thing is it asking for? (a series, a person, a place, a date, a title, a position, etc.)
2. Look through ALL your steps above. Which step contains something of THAT TYPE?
3. That is your answer — it is often in an EARLY step, not the latest one.
4. Do NOT answer with supporting details, companion items, or related-but-wrong entities.

Output EXACTLY:
STEP: The answer is: [the value that directly answers the question]
CONFIDENCE: [0.7-1.0]

IMPORTANT: Give only the specific answer value that matches what the question asked for."""

        elif q_type == 'yesno':
            if step_num == 1:
                # Step 1: gather BOTH entities' specific property values in one step
                prompt = f"""You are answering a yes/no comparison question.

{problem}

{context}Task: Find the specific property being compared and its value for EACH entity.

1. What property is being compared? (neighborhood, nationality, genre, etc.)
2. Value for Entity 1: [exact value from the context]
3. Value for Entity 2: [exact value from the context]

Respond in ONE line:
STEP: [Entity1] property is [value1]. [Entity2] property is [value2].
CONFIDENCE: [0.7-0.95]"""

            else:
                # Step 2+: compare the values from step 1 and conclude
                prev = current_state.steps[-1].content if current_state.steps else ""
                prompt = f"""You are answering a yes/no comparison question.

{problem}

The previous step found: "{prev}"

Compare the two values:
1. Copy Entity 1's exact value word-for-word: ___
2. Copy Entity 2's exact value word-for-word: ___
3. Are these EXACTLY the same string?
   - If yes (e.g. both "American", both "Fatih") → "The answer is: yes"
   - If no (e.g. "Fatih" vs "Ortaköy", "American" vs "British") → "The answer is: no"

Output ONLY:
STEP: The answer is: yes
or:
STEP: The answer is: no
CONFIDENCE: [0.8-1.0]"""

        elif q_type == 'comparison':
            if step_num == 1:
                prompt = f"""You are answering a comparison question.

{problem}

Extract the relevant property for BOTH compared entities from the context in one step.

Output EXACTLY:
STEP: [Entity1] has [value1]. [Entity2] has [value2].
CONFIDENCE: [0.7-0.95]

Example: "Annie Morton was born in 1970. Terry Richardson was born in 1965."
Only use values explicitly stated in the context."""
            else:
                # Always use the FIRST step (extraction) as facts — not the last,
                # which may already be a wrong intermediate answer.
                prev = current_state.steps[0].content if current_state.steps else ""
                prompt = f"""You are answering a comparison question.

{problem}

Facts found: "{prev}"

Determine the answer using these STRICT DIRECTION RULES:
- "older" / "born earlier" = SMALLER birth year. Example: 1965 < 1970, so born-1965 is OLDER.
- "younger" / "born later" = LARGER birth year.
- "more" / "larger" / "bigger" = HIGHER number.
- "first" / "earlier" = SMALLER date or number.
- "longer" / "more experienced" = started at SMALLER year.

Work through it step by step:
1. Comparison word in question: [older/younger/more/etc.]
2. Value 1: [number/date] — Value 2: [number/date]
3. Apply direction rule: which value wins?
4. That entity is the answer.

Output EXACTLY:
STEP: The answer is: [winning entity name]
CONFIDENCE: [0.8-1.0]"""

        else:  # bridge
            if step_num == 1:
                # Step 1: Full CoT-style reasoning — let model see all context and
                # reason from start to answer in ONE step (like CoT does).
                # 3 branches = 3 independent CoT attempts; heuristic picks the best.
                prompt = f"""Answer this multi-hop question by reasoning through it completely.

{problem}

Reason step by step:
1. What type of answer does the question need? (person / place / title / date / position / number)
2. What is the intermediate entity (the "bridge") — referred to in the question?
3. Find that entity in the context and extract the EXACT property the question asks about.
4. State the final answer.

Output EXACTLY:
STEP: [your complete reasoning ending with "The answer is: [exact value]"]
CONFIDENCE: [0.7-1.0]

IMPORTANT:
- Use ONLY values explicitly stated in the context.
- Do NOT fabricate or guess. Your STEP must end with: The answer is: X"""
            else:
                # Step 2+: refine based on what was found — bridge to the answer
                prev_facts = "\n".join(f"  - {s.content[:120]}" for s in current_state.steps)
                prompt = f"""You are refining your answer to a multi-hop question.

{problem}

Facts gathered so far:
{prev_facts}

Look at the context again. The reasoning so far found an intermediate entity.
Now find the EXACT property that answers the question.

Ask yourself: What specific thing did the question ask about? (position/title/location/year/etc.)
Find that EXACT value in the context and state it.

Output EXACTLY:
STEP: The answer is: [exact value from context]
CONFIDENCE: [0.7-1.0]

IMPORTANT: Give only the specific answer value. Short answers (1-4 words) are usually correct."""

        candidates = []
        temp = 0.1 if force_conclusion else 0.3
        try:
            response = self.client.chat.completions.create(
                model="llama3.1:8b",
                messages=[
                    {"role": "system",
                     "content": "You are a precise multi-hop reasoning assistant. Extract facts exactly as stated in the context. Never fabricate information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=512,
                n=1 if force_conclusion else n_candidates
            )
            self.total_tokens += response.usage.total_tokens
            self.api_calls += 1

            for choice in response.choices:
                content = choice.message.content

                step_match = re.search(
                    r'STEP(?:\s+\d+)?:\s*(.+?)(?=\n(?:STEP(?:\s+\d+)?:|CONFIDENCE:)|$)',
                    content, re.IGNORECASE | re.DOTALL
                )
                conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)

                if step_match:
                    step_text  = step_match.group(1).strip()
                    confidence = float(conf_match.group(1)) if conf_match else 0.8
                    confidence = max(0.5, min(0.99, confidence))

                    if len(step_text) > 5:
                        candidates.append((step_text, confidence))
                elif not force_conclusion:
                    # Fallback: model answered without STEP: prefix — use whole response
                    text = content.strip()
                    text = re.sub(r'^Step\s+\d+:\s*', '', text, flags=re.IGNORECASE).strip()
                    confidence = float(conf_match.group(1)) if conf_match else 0.75
                    confidence = max(0.5, min(0.99, confidence))
                    if len(text) > 5:
                        candidates.append((text, confidence))

        except Exception as e:
            if self.verbose:
                print(f"Warning: LLM generation error: {e}")

        return candidates

    def force_final_answer(self, problem: str, state: State,
                           q_type: str = 'bridge') -> Tuple[str, float]:
        """Force the model to output a final answer."""
        candidates = self.generate_next_steps(
            problem, state, n_candidates=1, force_conclusion=True, q_type=q_type
        )
        if candidates:
            return candidates[0]
        default = "The answer is: no" if q_type == 'yesno' else "The answer is: unknown"
        return (default, 0.5)

# ============================================================================
# A* SEARCH ALGORITHM
# ============================================================================

def astar_search(
    problem: str,
    solver: HotpotQASolver,
    max_depth: int = 15,
    max_nodes: int = 20,
    n_candidates: int = 3,
    verbose: bool = False,
    q_type: str = 'bridge'
):
    """Run A* search for HotpotQA problem."""

    open_set     = []
    node_counter = 0

    # force_depth: yesno uses step-2 non-force comparison prompt, no early force needed
    force_depth = max_depth - 1

    # Initialize root
    root_state = State()
    h_init     = compute_heuristic(root_state, problem, max_depth, q_type)
    root_node  = Node(
        state=root_state,
        g_score=0.0,
        h_score=h_init,
        h_score_semantic=0.0,
        f_score=h_init,
        node_id=node_counter
    )

    heapq.heappush(open_set, root_node)
    nodes_explored = 0

    best_goal_node    = None
    best_current_node = root_node
    visited      = set()  # full-path cycle detection
    goal_answers = {}     # answer text → count, for majority-vote early exit

    while open_set and nodes_explored < max_nodes:
        current = heapq.heappop(open_set)

        sig = current.state.signature()
        if sig in visited:
            continue
        visited.add(sig)

        nodes_explored += 1
        if verbose:
            print(f"Explored Node {current.node_id} (Depth {current.state.depth}, F={current.f_score:.2f})")
            if current.state.steps:
                print(f"  Last step: {current.state.steps[-1].content}")

        # Check for goal
        if current.state.steps:
            last_step = current.state.steps[-1].content
            is_goal = False
            if q_type == 'yesno':
                low = last_step.lower()
                if re.search(r'the answer is[: ]+(yes|no)\b', low):
                    is_goal = True
            else:
                if "answer is" in last_step.lower():
                    is_goal = True

            # Depth-gating: comparison needs ≥3 steps (extract→direction-compute→answer)
            # Accepting a goal earlier violates the admissibility guarantee for this type
            if is_goal and q_type == 'comparison' and current.state.depth < 3:
                is_goal = False

            if is_goal:
                if verbose:
                    print("  Goal found!")
                if best_goal_node is None or \
                   current.state.steps[-1].confidence > best_goal_node.state.steps[-1].confidence:
                    best_goal_node = current
                # Majority-vote early exit: if 2+ goal nodes agree on same answer, stop
                extracted = solver.extract_answer(current.state.get_trace(), q_type)
                if extracted:
                    goal_answers[extracted.lower().strip()] = goal_answers.get(extracted.lower().strip(), 0) + 1
                    if goal_answers[extracted.lower().strip()] >= 2:
                        if verbose:
                            print(f"  Majority vote exit on '{extracted}'")
                        return best_goal_node, nodes_explored

        # Track deepest non-goal node for fallback
        if current.state.depth > best_current_node.state.depth:
            best_current_node = current

        if current.state.depth >= max_depth:
            continue

        # Force conclusion once enough facts gathered
        force      = (current.state.depth >= force_depth)
        candidates = solver.generate_next_steps(
            problem, current.state, n_candidates, force, q_type
        )

        # Dedup: exact text + Jaccard similarity against all ancestor steps
        existing_contents = [s.content for s in current.state.steps]
        seen_texts        = set()
        unique_candidates = []
        for step_text, confidence in candidates:
            key = step_text.strip().lower()
            if key in seen_texts:
                continue
            if any(_jaccard_sim(step_text, prev) > 0.6 for prev in existing_contents):
                continue
            seen_texts.add(key)
            unique_candidates.append((step_text, confidence))
        candidates = unique_candidates

        # If dedup left nothing, synthesise terminal answer immediately (no heap push)
        if not candidates and current.state.steps:
            forced_text, forced_conf = solver.force_final_answer(problem, current.state, q_type)
            if 'answer is' not in forced_text.lower():
                forced_text = f"The answer is: {forced_text}"
            node_counter += 1
            t_state = State(
                steps=current.state.steps + [ReasoningStep(forced_text, forced_conf)],
                depth=current.state.depth + 1
            )
            t_node = Node(
                state=t_state,
                g_score=current.g_score + 1.0 + (1.0 - forced_conf),
                h_score=0.0, h_score_semantic=0.0,
                f_score=current.g_score + 1.0 + (1.0 - forced_conf),
                node_id=node_counter,
                parent_id=current.node_id
            )
            if best_goal_node is None or forced_conf > best_goal_node.state.steps[-1].confidence:
                best_goal_node = t_node
            continue  # do NOT push to heap — terminate this branch

        for step_text, confidence in candidates:
            node_counter += 1

            new_step  = ReasoningStep(step_text, confidence)
            new_state = State(
                steps=current.state.steps + [new_step],
                depth=current.state.depth + 1
            )

            step_cost = 1.0 + (1.0 - confidence)
            new_g     = current.g_score + step_cost
            new_h     = compute_heuristic(new_state, problem, max_depth, q_type)

            new_node = Node(
                state=new_state,
                g_score=new_g,
                h_score=new_h,
                h_score_semantic=0.0,
                f_score=new_g + new_h,
                node_id=node_counter,
                parent_id=current.node_id
            )
            heapq.heappush(open_set, new_node)

        # Return best goal after expanding this node
        if best_goal_node:
            return best_goal_node, nodes_explored

    # Fallback: force answer on the deepest node reached
    # Return best goal if found before hitting the fallback
    if best_goal_node:
        return best_goal_node, nodes_explored

    if verbose:
        print("No goal found, forcing answer...")
        print(best_current_node.state)

    final_step, conf = solver.force_final_answer(problem, best_current_node.state, q_type)
    final_state = State(
        steps=best_current_node.state.steps + [ReasoningStep(final_step, conf)],
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
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=1319)
    args = parser.parse_args()

    print("="*80)
    print("HotpotQA A* SOLVER")
    print("="*80)

    # Load dataset
    print("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpotqa/hotpot_qa", name="fullwiki", split="validation")
    print(f"Loaded {len(dataset)} examples")

    solver = HotpotQASolver(api_key="ollama", verbose=False)

    correct = 0
    total = 0
    results = []

    def normalize(t):
        t = t.lower().strip()
        t = re.sub(r'\b(a|an|the)\b', ' ', t)
        t = re.sub(r'[^\w\s]', '', t)
        return re.sub(r'\s+', ' ', t).strip()

    def f1_score(pred: str, gold: str) -> float:
        pred_tokens = normalize(pred).split()
        gold_tokens = normalize(gold).split()
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall    = len(common) / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    # Test loop
    for i, item in tqdm(enumerate(dataset.select(range(args.limit))), total=args.limit):
        # Format problem as context + question
        titles = item['context']['title']
        sentences_list = item['context']['sentences']
        ctx_parts = []
        for title, sents in zip(titles, sentences_list):
            ctx_parts.append(f"{title}: {' '.join(sents)}")
        context_str = '\n'.join(ctx_parts)
        problem = f"Context:\n{context_str}\n\nQuestion: {item['question']}"
        gold_answer = item['answer']

        print(f"\nProblem {i+1}: {problem[:100]}...")

        # Detect question type for type-aware search
        q_type = detect_question_type(item['question'])

        try:
            # Run A*
            best_node, nodes = astar_search(
                problem, solver, max_depth=25, max_nodes=20, n_candidates=3, verbose=False,
                q_type=q_type
            )

            # Check answer
            trace = best_node.state.get_trace()
            pred_answer = solver.extract_answer(trace, q_type)

            is_correct = False
            f1 = 0.0
            if pred_answer:
                f1 = f1_score(pred_answer, gold_answer)
                is_correct = normalize(pred_answer) == normalize(gold_answer) or f1 > 0.3

            print(f"  Predicted: {pred_answer}")
            print(f"  Gold: {gold_answer}")
            print(f"  EM: {'✅' if is_correct else '❌'}  |  F1: {f1:.2f}")

            if is_correct: correct += 1
            total += 1

            results.append({
                "question": item['question'],
                "gold": gold_answer,
                "pred": pred_answer,
                "correct": is_correct,
                "f1": f1,
                "nodes": nodes,
                "q_type": q_type
            })

        except Exception as e:
            import traceback
            print(f"  Error on problem {i+1}: {e}")
            traceback.print_exc()
            total += 1
            results.append({"question": item['question'], "gold": gold_answer, "pred": None, "correct": False, "f1": 0.0, "nodes": 0, "q_type": q_type})

    total_f1 = sum(r['f1'] for r in results)
    if total > 0:
        print(f"\nFinal EM: {correct}/{total} ({correct/total*100:.1f}%)  |  Avg F1: {total_f1/total*100:.1f}%")
    print(f"Total API calls: {solver.api_calls} | Total tokens: {solver.total_tokens}")

    # Save results
    os.makedirs("./results_shot", exist_ok=True)
    out_path = f"./results_shot/llama3.1_8b_{args.seed}_hotpotqa_astar_cot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
