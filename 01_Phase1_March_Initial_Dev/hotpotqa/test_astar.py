#!/usr/bin/env python3
"""
TEST FILE — HotpotQA A* Solver
Structured identically to gsm8k_astar reference. llama3.2:3b locally.
"""

import re
import heapq
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from openai import OpenAI
from datasets import load_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

LAMBDA = 0.12   # Weight for depth penalty (encourage efficiency)
MODEL  = "llama3.1:8b"
LIMIT  = 5

# Initialize Ollama client
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

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
    g_score: float       # Actual cost
    h_score: float       # Estimated cost to goal
    h_score_semantic: float
    f_score: float       # Total cost (g + h)
    node_id: int
    parent_id: Optional[int] = None

    def __lt__(self, other: "Node") -> bool:
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.node_id < other.node_id

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
    # Yes/No signals  (train.csv: ~6% of data, all comparison type)
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
    # also catch "were X and Y [adjective/noun]?" patterns (train.csv: comparison always yes/no)
    if re.search(r'(were|are|is|was)\s+\w[\w\s]+\s+and\s+\w[\w\s]+\s+(both|also|the same|known)', q):
        return 'yesno'
    # Comparison signals (train.csv: non-yesno comparison)
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
#   - max h(root) with LAMBDA=0.12, max_depth=25, max_complexity=4:
#     h(root) = (0.12*25 + 0.12*4) / 2 = (3.0 + 0.48) / 2 = 1.74
#   - 1.74 < 2.02  =>  ADMISSIBLE ✓

def h_remaining_steps(state: State, max_depth: int = 8) -> float:
    """Heuristic based on remaining depth budget."""
    remaining = max(0, max_depth - state.depth)
    return LAMBDA * remaining

def h_qa_complexity(state: State, question: str, q_type: str = 'bridge') -> float:
    """
    Type-aware complexity heuristic.
    Complexity calibrated so h never exceeds min 2-hop cost (admissible).
      yesno      : need 2 hops exactly  -> complexity starts at 2
      comparison : need 2 hops          -> complexity starts at 2
      bridge     : need 2+ hops         -> complexity starts at 2, +1 if chain keywords
    Decrements by 1 per depth to approach 0 at solution depth.
    """
    if q_type == 'yesno':
        complexity = 2  # always exactly 2 hops: extract property each entity, compare
    elif q_type == 'comparison':
        complexity = 2  # same structure as yesno but factual answer
    else:  # bridge
        q_lower = question.lower()
        complexity = 2
        if any(w in q_lower for w in ['who directed', 'who wrote', 'who played',
                                       'who founded', 'who created', 'who starred']):
            complexity += 1  # extra bridge step to identify person first
    remaining = max(0, complexity - state.depth)
    return LAMBDA * remaining

def compute_heuristic(state: State, problem: str, max_depth: int = 8,
                      q_type: str = 'bridge') -> float:
    """Compute combined admissible heuristic."""
    # Extract just the question from problem string for complexity calc
    q_match = re.search(r'Question:\s*(.+)$', problem, re.MULTILINE)
    question = q_match.group(1).strip() if q_match else problem
    h1 = h_remaining_steps(state, max_depth)
    h2 = h_qa_complexity(state, question, q_type)
    return (h1 + h2) / 2

# ============================================================================
# SIMILARITY HELPER  (Jaccard — filters near-duplicate steps)
# ============================================================================

def _jaccard_sim(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

# ============================================================================
# SOLVER LOGIC
# ============================================================================

class HotpotQASolver:
    def __init__(self, api_key: str = "ollama", verbose: bool = True):
        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        self.total_tokens = 0
        self.api_calls    = 0
        self.verbose      = verbose

    def extract_answer(self, trace: str, q_type: str = 'bridge') -> Optional[str]:
        """Extract answer, with yes/no normalization for yesno questions."""
        matches = re.findall(r'[Tt]he answer is[: ]+([^.\n]+)', trace)
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
            # Strip any "Step N: " prefix
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

        # Format history
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
                prompt = f"""You are answering a yes/no comparison question.

{problem}

{context}
Compare the property mentioned in the question step by step:
1. What specific property is being compared? (neighborhood, nationality, genre, etc.)
2. What is the value of that property for Entity 1? (exact value from step above)
3. What is the value of that property for Entity 2? (exact value from step above)
4. If the two values are the SAME → "The answer is: yes". If DIFFERENT → "The answer is: no".

Output EXACTLY one of:
STEP: The answer is: yes
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
            # Yes/No: gather BOTH entities' properties in one combined step,
            # then force conclusion at the next depth.
            prompt = f"""You are answering a yes/no comparison question.

{problem}

{context}STEP {step_num}: Find the relevant property for BOTH entities in ONE step.

RULES:
1. Identify the property being compared (neighborhood, nationality, genre, date, etc.).
2. Look up that property for the FIRST entity — cite the exact article.
3. Look up that property for the SECOND entity — cite the exact article.
4. State both facts together: "[Entity1] has [value1]. [Entity2] has [value2]."
5. Do NOT conclude yes/no here — just state the two facts.

Output EXACTLY:
STEP: [Fact about entity 1]. [Fact about entity 2].
CONFIDENCE: [0.6-0.95]

Step {step_num}:"""

        elif q_type == 'comparison':
            prompt = f"""You are answering a comparison question step-by-step.

{problem}

{context}STEP {step_num}: Extract one specific fact to compare.

RULES:
1. Step 1: Find the relevant property of the FIRST entity — cite the article.
2. Step 2: Find the relevant property of the SECOND entity — cite the article.
3. Step 3: Compare and write "The answer is: [winning/correct entity or value]".
4. Extract EXACT values (dates, numbers, names) — do not paraphrase.

Output EXACTLY:
STEP: [direct fact from context OR "The answer is: X"]
CONFIDENCE: [0.6-0.95]

Step {step_num}:"""

        else:  # bridge
            prompt = f"""You are answering a multi-hop bridge question step-by-step.

{problem}

{context}STEP {step_num}: Find the next fact in the reasoning chain.

RULES:
1. Step 1: Identify the intermediate entity (person/place/thing) referenced in the question.
2. Step 2: Look up the specific property of that entity in the context.
3. Write "The answer is: [value]" when you can fully answer the question.
4. Cite the article: "According to [Article Title], ..."
5. Extract EXACT values — do not rephrase or summarise.

Output EXACTLY:
STEP: [direct fact from context OR "The answer is: X"]
CONFIDENCE: [0.6-0.95]

Step {step_num}:"""

        candidates = []
        temp = 0.1 if force_conclusion else 0.3
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a precise multi-hop reasoning assistant. Extract facts exactly as stated in the context. Never fabricate information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=256,
                n=1 if force_conclusion else n_candidates
            )
            self.total_tokens += response.usage.total_tokens
            self.api_calls    += 1

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

        except Exception as e:
            if self.verbose:
                print(f"Warning: LLM generation error: {e}")

        print(candidates)
        return candidates

    def force_final_answer(self, problem: str, state: State,
                           q_type: str = 'bridge') -> Tuple[str, float]:
        """Force the model to output a final answer."""
        candidates = self.generate_next_steps(
            problem, state, n_candidates=1, force_conclusion=True, q_type=q_type
        )
        if candidates:
            return candidates[0]
        default = "The answer is: yes" if q_type == 'yesno' else "The answer is: unknown"
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

    # Yes/no: step 1 gathers both entities' facts, step 2 forces comparison
    force_depth = 1 if q_type == 'yesno' else max_depth - 1

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
    visited         = set()   # cycle / revisit detection (full path signature)

    while open_set and nodes_explored < max_nodes:
        current = heapq.heappop(open_set)

        # Skip already-explored states (same reasoning path)
        sig = current.state.signature()
        if sig in visited:
            continue
        visited.add(sig)

        nodes_explored += 1

        if verbose:
            print(f"Explored Node {current.node_id} (Depth {current.state.depth}, F={current.f_score:.2f})")
            if current.state.steps:
                print(f"  Last step: {current.state.steps[-1].content}")

        # Check for goal: "The answer is: X" (or yes/no)
        if current.state.steps:
            last_step = current.state.steps[-1].content
            is_goal = False
            if q_type == 'yesno':
                # Accept only explicit yes/no answers
                low = last_step.lower()
                if re.search(r'the answer is[: ]+(yes|no)\b', low):
                    is_goal = True
            else:
                if "answer is" in last_step.lower():
                    is_goal = True

            if is_goal:
                if verbose:
                    print("  Goal found!")
                if best_goal_node is None or \
                   current.state.steps[-1].confidence > best_goal_node.state.steps[-1].confidence:
                    best_goal_node = current
                # Early exit only for very high confidence AND not too shallow
                if current.state.steps[-1].confidence > 0.95 and current.state.depth >= 2:
                    return current, nodes_explored

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

        # --- Dedup: exact text + Jaccard similarity against all steps in current path ---
        existing_contents = [s.content for s in current.state.steps]
        seen_texts        = set()
        unique_candidates = []
        for step_text, confidence in candidates:
            key = step_text.strip().lower()
            if key in seen_texts:
                continue
            # Drop if too similar to any ancestor step (prevents paraphrase loops)
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

            step_cost = 1.0 + (1.0 - confidence)   # lower cost = higher confidence
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

        # Return best goal found: prefer highest confidence (not lowest g)
        # Rationale: in QA, confident late answers beat cheap early hallucinations
        if best_goal_node:
            return best_goal_node, nodes_explored

    # Fallback: force answer on the deepest node reached
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
# EVALUATION HELPERS
# ============================================================================

def normalize(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r'\b(a|an|the)\b', ' ', t)
    t = re.sub(r'[^\w\s]', '', t)
    return re.sub(r'\s+', ' ', t).strip()

def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 — HotpotQA official metric."""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common    = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print(f"TEST: HotpotQA A* — model={MODEL}, limit={LIMIT}")
    print("=" * 60)

    ds     = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    solver = HotpotQASolver(api_key="ollama", verbose=True)

    correct = 0
    results = []

    for i, item in enumerate(ds):
        if i >= LIMIT:
            break

        # Build context string
        titles         = item['context']['title']
        sentences_list = item['context']['sentences']
        ctx_parts      = [f"{t}: {' '.join(s)}" for t, s in zip(titles, sentences_list)]
        context_str    = '\n'.join(ctx_parts)
        problem        = f"Context:\n{context_str}\n\nQuestion: {item['question']}"
        gold_answer    = item['answer']

        print(f"\n{'='*60}")
        q_type = detect_question_type(item['question'])
        print(f"Problem {i+1} [{q_type}]: {item['question']}")
        print(f"Gold: {gold_answer}")

        try:
            best_node, nodes = astar_search(
                problem, solver, max_depth=25, max_nodes=20,
                n_candidates=3, verbose=True, q_type=q_type
            )

            trace       = best_node.state.get_trace()
            pred_answer = solver.extract_answer(trace, q_type)

            f1         = f1_score(pred_answer, gold_answer) if pred_answer else 0.0
            is_correct = (normalize(pred_answer) == normalize(gold_answer) or f1 > 0.3) if pred_answer else False

            print(f"\nFull Trace:\n{trace}")
            print(f"\nPredicted: {pred_answer}")
            print(f"Gold:      {gold_answer}")
            print(f"EM: {'✅' if is_correct else '❌'}  |  F1: {f1:.2f}")

            if is_correct:
                correct += 1

            results.append({
                "question": item['question'],
                "gold":     gold_answer,
                "pred":     pred_answer,
                "correct":  is_correct,
                "f1":       f1,
                "nodes":    nodes,
                "q_type":   q_type
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()

    total  = len(results)
    avg_f1 = sum(r['f1'] for r in results) / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"Test Accuracy (EM+F1>0.3): {correct}/{LIMIT} ({correct/LIMIT*100:.1f}%)")
    print(f"Avg F1: {avg_f1:.1f}%")
    print(f"Total API calls: {solver.api_calls} | Total tokens: {solver.total_tokens}")

if __name__ == "__main__":
    main()
