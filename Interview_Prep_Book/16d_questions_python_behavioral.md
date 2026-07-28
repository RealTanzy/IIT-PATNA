# Chapter 16d: Python Engineering & Behavioral Interview Questions (Q171-210)

---

## 16.9 Python & Engineering (Q171-195)

### Q171: What is asyncio and when would you use it?

**Answer:** asyncio is Python's built-in library for writing concurrent code using async/await syntax. It uses a single-threaded event loop to handle multiple I/O-bound operations concurrently (not in parallel). Use it when your program spends most of its time waiting for I/O (API calls, database queries, file reads) rather than doing CPU-heavy computation. It's NOT suitable for CPU-bound work — use ProcessPool for that.

**In Practice:** SSV RAG uses `asyncio.gather()` to query Jira SWF and Jira MCP in parallel, cutting latency by ~1s per request.

---

### Q172: What's the difference between ProcessPoolExecutor and ThreadPoolExecutor?

**Answer:** ThreadPoolExecutor runs tasks in separate threads within the same process — shares memory, subject to the GIL (Global Interpreter Lock), best for I/O-bound tasks. ProcessPoolExecutor spawns separate processes — each has its own GIL, truly parallel, best for CPU-bound tasks. Threads are lightweight (microseconds to create), processes are heavy (milliseconds + memory copy). Use threads for API calls, processes for data processing or model inference.

**In Practice:** SSV RAG uses ThreadPoolExecutor(max_workers=2) to run BM25 and ANN retrieval in parallel (both are I/O-bound LanceDB queries).

---

### Q173: Explain the decorator pattern in Python. Give a real example.

**Answer:** A decorator is a function that wraps another function, adding behavior without modifying the original. It uses the `@decorator` syntax. Common uses: logging, caching, retry logic, authentication, and plugin registration. The key insight is that decorators are just syntactic sugar for `func = decorator(func)`.

**In Practice:** SPOT CHECK V2 uses `@register(N)` decorators for auto-discovery of checkpoint evaluators — each evaluator class is decorated with its checkpoint number, and the registry automatically collects all of them at import time.

```python
# Registry pattern with decorators
_REGISTRY = {}

def register(checkpoint_number):
    def decorator(cls):
        _REGISTRY[checkpoint_number] = cls
        return cls
    return decorator

@register(1)
class Q01TestStrategy(CheckpointEvaluator):
    ...

@register(2)
class Q02Consistency(CheckpointEvaluator):
    ...
```

---

### Q174: What is Pydantic v2 and why use it for AI systems?

**Answer:** Pydantic is a data validation library that uses Python type hints to define data models with automatic validation, serialization, and documentation. V2 is a complete rewrite in Rust (10-50× faster). For AI systems, it enforces structured LLM outputs — you define a Pydantic model and the LLM must return JSON matching that schema. If it doesn't, validation fails and you can retry.

**In Practice:** SPOT CHECK uses Pydantic v2 for all domain models (SubPointResult, CheckpointResult, FollowUpQuestion, AssessmentSession) ensuring type safety throughout the evaluation pipeline.

---

### Q175: How do you implement retry logic with exponential backoff?

**Answer:** Exponential backoff increases wait time between retries: 1s, 2s, 4s, 8s... This prevents thundering herd problems when a service recovers. In Python, use the `tenacity` library: `@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(3), retry=retry_if_exception_type((TimeoutError, HTTPError)))`. Always set a max retry count and max wait time to avoid infinite loops.

**In Practice:** iFAST uses tenacity with exponential backoff (2s-30s) for the GenAI Nexus API, retrying on 429 (rate limit), 502, and 503 errors.

---

### Q176: Explain the Adapter pattern. When do you use it?

**Answer:** The Adapter pattern provides a unified interface for different implementations. Define an abstract base class with common methods, then create concrete adapters for each variant. Use it when you need to support multiple data formats, APIs, or backends through a single interface. The client code only talks to the abstract interface — never to specific adapters.

**In Practice:** SPOT CHECK uses the Adapter pattern for document ingestion — PDFAdapter, DOCXAdapter, HTMLAdapter, ExcelAdapter, CSVAdapter, ImageAdapter, ConfluenceAdapter — all implementing `can_handle()` + `extract()`. Adding a new format = just writing a new adapter.

---

### Q177: What is connection pooling and why does it matter?

**Answer:** Connection pooling reuses existing TCP/TLS connections instead of creating new ones per request. A TLS handshake takes ~100-300ms; with pooling, subsequent requests to the same host skip this entirely. In Python, use `requests.Session()` with `HTTPAdapter(pool_connections=10, pool_maxsize=10)`. This is critical for services making many API calls to the same endpoint.

**In Practice:** SSV RAG maintains persistent connection pools to Jira SWF and Jira MCP, saving ~1-2s per query cycle. The sessions are cached as Streamlit `@st.cache_resource` singletons.

---

### Q178: How do you handle streaming responses from an LLM?

**Answer:** Streaming delivers tokens as they're generated (SSE — Server-Sent Events). The client reads chunks incrementally rather than waiting for the full response. This improves perceived latency dramatically (first token in ~800ms vs 5-8s for full response). Implementation: iterate over the response stream, yield tokens to the frontend, buffer for partial JSON parsing. Handle connection drops gracefully.

**In Practice:** SSV RAG streams Claude Opus responses via Amazon EventStream binary protocol AND SSE (handles both formats). The Streamlit frontend renders tokens live with auto-scroll.

---

### Q179: What is a singleton pattern and when do you use it in ML systems?

**Answer:** A singleton ensures only one instance of a class exists. In ML systems, use it for expensive resources that should be loaded once: embedding models, reranker models, database connections, configuration objects. In Streamlit, `@st.cache_resource` achieves this. Without singletons, each request might reload a 400MB model — catastrophic for latency.

**In Practice:** SSV RAG uses singletons for the embedding model, cross-encoder reranker, LanceDB connection, and DataFetcher — all via `@st.cache_resource`. This was Fix #1 in the 22s→4s optimization.

---

### Q180: How do you design an evaluation framework for an AI system?

**Answer:** (1) Define categories of queries (by source, difficulty, intent). (2) Write 100+ test queries with expected behavior (not exact answers — check if answer is substantive). (3) Run all queries programmatically, record timing + outputs. (4) Measure: answer rate, guardrail compliance, context hit rate, latency percentiles. (5) Re-run after every change to detect regressions. Store results as JSON for comparison.

**In Practice:** SSV RAG has a 100-query benchmark: 25 SWF + 25 MCP + 25 Confluence + 25 Mixed/Guardrail queries. Results: 77% answer rate, 100% guardrail compliance.

---

### Q181: Explain structured logging vs print statements.

**Answer:** Structured logging outputs machine-parseable records (JSON) with consistent fields (timestamp, level, module, message, context). Print statements are unstructured, can't be filtered/searched/aggregated. Use libraries like `structlog` or `loguru`. Structured logs enable: searching by request_id, filtering by severity, aggregating error rates, and integrating with monitoring systems (Grafana, Datadog).

**In Practice:** SSV RAG uses Loguru with file rotation for all pipeline logging. Each log includes module, timestamp, and request context.

---

### Q182: What's the difference between FastAPI and Streamlit? When use which?

**Answer:** FastAPI is a web framework for building APIs (REST endpoints, OpenAPI docs, async support, Pydantic validation). Streamlit is a UI framework for building data apps (widgets, charts, session state, instant prototyping). Use FastAPI when building backends that other services consume. Use Streamlit when building internal tools where the user IS the developer/analyst. They can be combined: Streamlit frontend → FastAPI backend.

**In Practice:** SSV RAG and SPOT CHECK both use Streamlit for the frontend (rapid iteration, no frontend engineers needed). iFAST uses Streamlit for the web interface but also has a PySide6/Qt desktop application.

---

### Q183: How do you manage secrets and configuration in production?

**Answer:** Never hardcode secrets. Use environment variables loaded via `python-dotenv` or a secrets manager (AWS Secrets Manager, HashiCorp Vault). Configuration hierarchy: defaults in code → .env file → environment variables (highest priority). Keep a `.env.example` with placeholder values in version control. Validate required vars at startup — fail fast if missing.

**In Practice:** All Mercedes projects use `.env` files with `python-dotenv`. SPOT CHECK has a `Settings` singleton (Pydantic BaseSettings) that validates all config at startup.

---

### Q184: How do you handle rate limiting from LLM APIs?

**Answer:** (1) Respect `Retry-After` headers — back off for the specified duration. (2) Implement exponential backoff with jitter. (3) Use semaphores to limit concurrent requests (`asyncio.Semaphore(5)`). (4) Cache responses to avoid redundant calls. (5) Have a fallback model — if primary is rate-limited, switch to secondary. (6) Monitor token usage to stay within quotas.

**In Practice:** iFAST and SPOT CHECK both implement cross-model fallback — if Claude rate-limits, switch to the other model's API key. SSV RAG's semantic cache eliminates ~30% of LLM calls for repeated queries.

---

### Q185: Explain the difference between unit tests, integration tests, and evaluation tests for AI systems.

**Answer:** Unit tests: test individual functions in isolation (does `extract_features()` return 14 values?). Integration tests: test components together (does the pipeline retrieve → rerank → generate without errors?). Evaluation tests: measure output *quality* (does the system answer correctly?). For AI systems, evaluation tests are most important but hardest — you need a benchmark dataset and quality metrics, not just "does it run."

**In Practice:** SSV RAG has all three: unit tests for retriever/embedder, integration tests for the full pipeline, and a 100-query evaluation benchmark measuring answer rate and latency.

---

### Q186: What is semantic chunking and why does it matter?

**Answer:** Semantic chunking splits documents at natural boundaries (sentences, paragraphs, sections) rather than at arbitrary character counts. This preserves meaning within each chunk — a chunk about "team EDS progress" won't be split mid-sentence. Implementation: split at sentence boundaries (`.!?`), merge sentences until chunk exceeds max size, add overlap for context continuity.

**In Practice:** SSV RAG uses table-aware semantic chunking — each Confluence table row becomes its own chunk with column headers injected, so retrieval for "EDS in CW24" returns exactly one relevant chunk.

---

### Q187: How do you implement a priority queue in Python?

**Answer:** Use `heapq` module — it provides a min-heap. Elements are tuples where the first element is the priority. `heapq.heappush(heap, (priority, item))` inserts in O(log n), `heapq.heappop(heap)` removes minimum in O(log n). For custom objects, implement `__lt__` for comparison. Common gotcha: heapq is a MIN-heap — for max-heap, negate priorities.

**In Practice:** The MTP thesis A* implementation uses `heapq` as the open set, with f_score as priority. Tie-breaking uses node_id (earlier nodes expanded first).

```python
@dataclass
class Node:
    f_score: float
    node_id: int
    
    def __lt__(self, other):
        if abs(self.f_score - other.f_score) > 1e-9:
            return self.f_score < other.f_score
        return self.node_id < other.node_id
```

---

### Q188: What is SHA-256 hashing and where do you use it in AI systems?

**Answer:** SHA-256 produces a fixed-length 256-bit hash from arbitrary input. It's deterministic (same input → same hash), collision-resistant (practically impossible to find two inputs with the same hash), and one-way (can't reverse). In AI systems: deduplicate documents (hash content → detect duplicates), cache keys (hash prompts for LLM cache), and verify data integrity.

**In Practice:** SSV RAG uses SHA-256 for document deduplication during ingestion (skip re-embedding unchanged docs) and MD5 for LLM disk cache keys (hash the full prompt → look up cached response).

---

### Q189: How does `@st.cache_resource` differ from `@st.cache_data`?

**Answer:** `@st.cache_resource` caches the object itself (shared across all users/sessions) — use for ML models, DB connections, heavy singletons. `@st.cache_data` caches the return value (serialized copy per call signature) — use for data fetches, computations. Key difference: cache_resource returns the SAME object (mutations visible everywhere); cache_data returns a COPY (mutations don't persist).

**In Practice:** SSV RAG uses `@st.cache_resource` for embedding model + reranker + LanceDB connection (loaded once, shared), and `@st.cache_data(ttl=300)` for retrieval results (cached 5 min, immutable).

---

### Q190: How do you implement file-based IPC (inter-process communication)?

**Answer:** When processes can't share memory (e.g., background subprocess + main app), use files as communication channels. Write status/progress to a JSON file from the producer; poll the file from the consumer (every 5s). Use atomic writes (write to temp file → rename) to prevent partial reads. Add timestamps to detect staleness.

**In Practice:** iFAST uses file-based IPC for real-time progress — the background data generation subprocess writes progress to a status file, and the Streamlit frontend polls it every 5 seconds to display live updates.

---

### Q191: What's the GIL and how does it affect ML workloads?

**Answer:** The Global Interpreter Lock (GIL) prevents multiple threads from executing Python bytecode simultaneously. This means ThreadPoolExecutor doesn't give true parallelism for CPU-bound code. However, the GIL is released during I/O operations (network calls, file reads) and during calls to C extensions (NumPy, PyTorch). So: threads work for I/O, processes needed for pure Python CPU work.

**In Practice:** SSV RAG uses threads for parallel API calls (GIL released during HTTP I/O). If we needed parallel embedding computation, we'd use ProcessPool (but the model already uses C/CUDA internally, releasing the GIL).

---

### Q192: How do you implement graceful degradation in an AI system?

**Answer:** Design the system to provide partial results rather than failing completely. Strategy: (1) Try primary path → on failure, try fallback. (2) Return cached/stale data with a "may be outdated" warning. (3) Answer with partial evidence rather than refusing entirely. (4) Timeout handling: return whatever you have after N seconds.

**In Practice:** iFAST generates a verdict using fast JSON evidence immediately (<60s for pre-generated data), while visual reports generate in the background (10+ min). Users get actionable results instantly; rich reports arrive asynchronously.

---

### Q193: Explain the Observer/Callback pattern for live UI updates.

**Answer:** The Observer pattern lets an object notify multiple observers when its state changes. In AI systems, use callbacks to update the UI during long-running operations. Pass a `thinking_callback` function to your pipeline; the pipeline calls it with progress updates. The UI renders these in real-time without polling.

**In Practice:** SPOT CHECK passes a `thinking_callback` to the assessment pipeline — as each sub-point is evaluated, the callback fires and the Streamlit UI shows "Evaluating Q3 sub-point (c)..." in real-time.

---

### Q194: How do you handle multimodal inputs (text + images)?

**Answer:** For LLMs with vision capabilities (Claude, GPT-4V, Gemini): encode image as base64, include in the messages array alongside text. Format: `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}`. For document images: extract via PyMuPDF (PDFs) or python-docx (Word), then send to vision API for OCR/description.

**In Practice:** SPOT CHECK extracts embedded images from PDFs and DOCX files, sends them to Claude's Vision API for content description, and includes the descriptions as text chunks in the RAG pipeline.

---

### Q195: What is delta sync and why is it important?

**Answer:** Delta sync only processes documents that changed since the last sync — checking `updated_at` timestamps or content hashes. Without delta sync, every sync re-embeds ALL documents (expensive: model inference for every chunk). With it, only modified docs get re-embedded. Implementation: store last-sync timestamp per document, compare on next run, skip unchanged.

**In Practice:** SSV RAG's KBSyncManager runs every 30 minutes but only re-embeds Confluence pages where `updated_at` has changed — saving ~90% of embedding computation.

---

## 16.10 Behavioral / Project Deep-Dives (Q196-210)

### Q196: Walk me through the iFAST system end to end.

**Situation:** At Mercedes-Benz, when automotive software signals fail during testing, engineers spend 4-8 hours manually investigating root causes across git diffs, Simulink diagrams, and spreadsheets. With 50+ failures per week, the backlog grew faster than the team could handle.

**Task:** Build an autonomous system that performs the entire investigation automatically — from failure report to cited root-cause verdict.

**Action:** I architected a single-agent system (Claude via AWS Bedrock) with 14 MCP tools communicating over JSON-RPC 2.0. The agent autonomously decides which tools to call: graph traversal (BFS over 1.6M edges), source code analysis, git diff comparison, configuration checks, and binary comparisons. Evidence from 8 independent sources is fused using calibrated Bayesian Log-Likelihood Ratios — each source contributes a weighted score, and the final verdict is P(CodeChange|Evidence) = sigmoid(Σ LLR_i).

**Result:** 54.8% time savings, 97% accuracy on root-cause classification, 75% false positive elimination via signal-level verification. Adopted by 2+ teams beyond the original scope.

---

### Q197: How does your Bayesian scoring work? Explain LLR.

**Situation:** The failure analysis system needs to combine evidence from 8 different verification sources (git diff, source code parser, configuration check, visual diff, etc.) into a single confidence score.

**Task:** Design a principled method to fuse heterogeneous evidence — not heuristic "if X then Y" rules, but mathematically grounded scoring.

**Action:** I implemented Log-Likelihood Ratios (LLR). Each evidence source contributes a pre-calibrated LLR: positive values indicate evidence of a code change (+3.5 for visual structural diff, +3.0 for deleted interface), negative values indicate evidence AGAINST a change (-2.5 for signal confirmed unchanged, -4.0 for all modules unchanged). The total evidence is the sum: `total_LLR = Σ LLR_i`. Final probability: `P = sigmoid(total_LLR)`. Decision rule: LLR > 0 with sufficient magnitude → "Code change caused failure"; LLR < 0 → "Inconclusive / not code-related."

**Result:** This eliminated 75% of false positives because contradicting evidence (signal unchanged: -2.5) mathematically cancels positive signals (module changed: +2.5). Engineers trust the system because every decision shows its evidence trail with quantified weights.

---

### Q198: Explain your A* admissibility proof step by step.

**Situation:** For the TMLR paper, reviewers expect formal guarantees that A* search finds the optimal reasoning path — not just empirical results.

**Task:** Prove that the entity-coverage + depth heuristic is admissible (never overestimates true remaining cost) and consistent (satisfies the triangle inequality).

**Action:** 
1. Defined step cost: c(s→s') = 1.0 + (1.0 - confidence) ≥ 1.01 (confidence capped at 0.99)
2. Established minimum path: root→goal requires ≥ 2 hops (multi-hop QA), so h*(root) ≥ 2 × 1.01 = 2.02
3. Computed max heuristic at each depth:
   - d=0: h ≤ 2×0.3 + 1×0.3 = 0.9 < 2.02 ✓
   - d=1: h ≤ 1×0.3 + 1×0.3 = 0.6 < 1.01 ✓
   - d≥2: h ≤ 0×0.3 + 1×0.3 = 0.3 < 1.01 ✓
4. Proved consistency: max|Δh| = 0.6 < 1.01 = min(c) ✓

**Result:** A* with graph search is proven optimal and complete. Published in the TMLR submission — the formal proof is in the paper's appendix.

---

### Q199: How did you optimize latency from 22s to 4s?

**Situation:** The SSV RAG chatbot took 22 seconds per query — unacceptable for real-time use by the validation team.

**Task:** Identify root causes and fix them systematically, with before/after measurements for each fix.

**Action:** I profiled the pipeline and found 5 independent bottlenecks:
1. **Model reloading per request** (~5s): embedding model + reranker loaded fresh each time. Fix: `@st.cache_resource` singletons.
2. **New HTTP connections per Jira call** (~1-2s): TLS handshake on every request. Fix: persistent connection pools (HTTPAdapter, pool_size=10).
3. **Sequential Jira queries** (~1s): SWF and MCP queried one after another. Fix: `ThreadPoolExecutor` parallel execution.
4. **16KB context to LLM** (~8-12s LLM generation time): too much context = slow generation. Fix: context compression to 15K chars with deduplication and sentence-boundary truncation.
5. **No caching** (full cost every time): identical queries hit the full pipeline. Fix: semantic cache (cosine 0.92) + LLM disk cache (MD5 hash, 30-min TTL).

**Result:** 22s → 4s median latency (5.5× improvement). Each fix is independently measurable and reversible.

---

### Q200: What was the hardest technical challenge you solved?

**Situation:** In iFAST, the existing dependency data came from a manual GUI tool (Movetto) that traced signal paths through SVG block diagrams — 55MB of static files, outdated within days, and 35% incomplete.

**Task:** Replace this brittle manual process with a fully automated, version-specific pipeline that's always fresh and complete.

**Action:** I reverse-engineered the naming conventions of auto-generated C source code (TargetLink pattern: `Rte_IWrite_{Module}_P_{Signal}_{Signal}`). Built a regex parser that extracts the same dependency information directly from source code at any git tag. The parser processes 1,128 modules across 3 repositories in 86 seconds, extracting 1.6M+ dependency edges. Results are cached permanently per git tag (deterministic — same tag always produces same graph).

**Result:** 35% more signal coverage, zero staleness, zero junk entries. Validated: byte-for-byte identical outputs when compared against the manual process on known-good data. The key insight was that auto-generated code follows strict naming patterns — regex is not "hacky" here, it's the mathematically correct parser for a regular language.

---

### Q201: How did you handle conflicting evidence in iFAST?

**Situation:** One source says "module changed" (+2.5 LLR), another says "specific signal is unchanged" (-2.5 LLR). Which do you trust?

**Task:** Design a system that handles contradictions gracefully rather than producing false positives.

**Action:** The Bayesian framework handles this naturally. Each source contributes independently to the total LLR. When sources contradict, their scores cancel: +2.5 + (-2.5) = 0, which produces P = sigmoid(0) = 0.5 = maximum uncertainty. The decision rule: if total_LLR ≤ 0, output "INCONCLUSIVE" regardless of individual positive signals. Additionally, cross-validation detects agreement patterns: TRIPLE confirmed (3+ sources agree) or contradiction flagged.

**Result:** Zero false positives on the "known unchanged" benchmark. The system correctly identifies when it doesn't have enough evidence rather than guessing.

---

### Q202: Tell me about a time you had to replace a manual process.

**Situation:** The Mercedes SSV team's dependency data was exported manually from a GUI tool, took 5-30 minutes, required human intervention, and became stale within days.

**Task:** Replace it with an automated, always-fresh alternative without losing accuracy.

**Action:** I studied the output format, identified the source-of-truth (auto-generated C code in git), reverse-engineered the naming conventions, built a regex-based parser that runs against any git tag, and validated the output against known-good manual exports. Deployed as part of iFAST — runs automatically when new data is needed.

**Result:** 86-second automated build (vs 30-min manual), 35% more complete, always version-specific (no staleness), zero human intervention required.

---

### Q203: How does your topology router beat LLM-based routers?

**Situation:** LLM-based routers (Critic, Judge, Semantic Entropy) require running the LLM at inference time — adding latency and cost. They also suffer from the Fluency-Correctness Asymmetry.

**Task:** Build a router that's faster (zero LLM calls), cheaper, and more accurate.

**Action:** I extracted 14 structural features from the problem text (word count, logical connectives, hop indicators, branching coefficient, etc.) and trained an XGBoost classifier on disagreement cases. The key insight: problem STRUCTURE predicts method success better than LLM confidence, because 38% of confident (fluent) traces are actually wrong. The topology router makes decisions in microseconds (feature extraction + classifier predict) vs seconds for LLM-based routing.

**Result:** Wins on 14/18 dataset-scale combinations. 40% gap recovery overall, 95% accuracy on CodeContests. The single strongest feature (Branching Coefficient = ctx_sents × hop / q_len) predicts method success at R²=0.94.

---

### Q204: Design the SPOT CHECK system from scratch.

**Situation:** An interviewer asks you to design an automated compliance assessment system.

**Task:** Explain the architecture, key decisions, and tradeoffs.

**Action:** I'd design it as a multi-agent pipeline:
1. **Document Ingestion:** Adapter pattern for multiple formats (PDF/DOCX/Excel/HTML). Each adapter extracts text + structure. Chunk semantically (section-aware, table-aware, 500 tokens max with 50 overlap).
2. **Vector Store:** Per-session ChromaDB (isolate each assessment's evidence). Embed with all-MiniLM-L6-v2.
3. **Evaluation Engine:** Plugin architecture — each compliance checkpoint is a registered evaluator class. For each sub-point: retrieve top-5 relevant chunks → ask Assessor Agent to judge (found/partial/missing) with mandatory evidence citation.
4. **Scoring:** Hierarchical — sub-point (0/0.5/1) → checkpoint (average) → overall (average of averages). Thresholds map to Yes/Partial/No.
5. **Output:** Excel template fill + PDF report + JSON evidence log.
6. **Anti-hallucination:** System prompts enforce "blank > wrong" policy. Every finding requires evidence_text + evidence_source.

**Result:** Replaces 2-3 day manual audits with sub-hour automated first pass. 60+ evaluation dimensions covered.

---

### Q205: What would you improve about your RAG system?

**Situation:** Reflecting honestly on limitations.

**Task:** Show self-awareness and technical depth.

**Action:** Three improvements I'd prioritize:
1. **Multi-turn reasoning:** Current system is single-shot. For complex queries ("compare CW20-24 trends AND suggest improvements"), I'd add a planning step that decomposes into sub-queries, retrieves for each, then synthesizes.
2. **Adaptive retrieval depth:** Currently fixed top_k=5 for all queries. Simple factual queries need 1-2 chunks; comparative queries need 10+. I'd use intent classification to set dynamic retrieval depth.
3. **Automated evaluation regression:** Currently manual — I'd integrate the 100-query benchmark into CI/CD so every code change is automatically tested against quality metrics.

**Result:** These would push answer rate from 77% toward 85%+ and catch quality regressions before deployment.

---

### Q206: Tell me about a time you disagreed with a team decision.

**Situation:** At Mercedes, the initial plan was to use a multi-agent architecture for iFAST (separate agents for graph traversal, code analysis, evidence fusion).

**Task:** I believed a single agent with multiple tools would be better for root-cause analysis.

**Action:** I argued that root-cause analysis requires a single coherent reasoning chain — the agent needs to see ALL evidence together to weigh contradictions. Multi-agent would fragment the context. I built a prototype of both approaches: the single-agent version produced more coherent verdicts because it could reason about cross-source contradictions in one context window.

**Result:** Team agreed after seeing the prototype comparison. The single-agent architecture achieved 97% accuracy. The key learning: multi-agent is great for independent subtasks, but single-agent is better when reasoning requires holistic context.

---

### Q207: How do you prioritize when you have multiple projects?

**Situation:** At Mercedes, I simultaneously managed iFAST (primary), RAG chatbot (team dependency), and SPOT CHECK (exploratory).

**Task:** Deliver all three without dropping quality.

**Action:** I prioritized by impact × urgency: iFAST was daily-use (highest impact, active users waiting), RAG chatbot had specific team requests (high urgency), SPOT CHECK was exploratory (lower urgency but high strategic value). I spent 60% on iFAST, 25% on RAG, 15% on SPOT CHECK. When blockers hit one project (waiting for data), I'd context-switch to another rather than idle.

**Result:** All three shipped. iFAST adopted by 2+ teams, RAG achieved 77% answer rate, SPOT CHECK V2 architecture completed.

---

### Q208: Describe a time you had to learn something new quickly.

**Situation:** When I started at Mercedes, I had never worked with automotive signal data, AUTOSAR architecture, or ASPICE compliance standards.

**Task:** Become productive within 2 weeks despite zero domain knowledge.

**Action:** I focused on understanding the INTERFACES not the domain — "what data flows where" rather than "what does this signal mean." I mapped the system as a graph (modules → signals → dependencies) which is domain-agnostic. For ASPICE, I read the standard as a rubric (checkpoints = rules, sub-points = criteria) rather than trying to understand all of automotive quality theory.

**Result:** First working prototype of iFAST within 3 weeks. The key insight: good software engineering principles (abstraction, interfaces, graphs) transfer across domains — you don't need to become a domain expert to build domain-expert tools.

---

### Q209: What's your biggest technical failure and what did you learn?

**Situation:** In the MTP thesis, the initial A* implementation on HotpotQA achieved only 44.6% accuracy — WORSE than the 76.8% CoT baseline. The entire approach seemed to fail.

**Task:** Diagnose why search was losing to a single-pass baseline.

**Action:** I did a root cause analysis and found 5 bugs:
1. No real branching (Ollama ignores n>1 → search was just linear CoT)
2. Premature exit (returned first goal, never explored alternatives)
3. Rigid prompts (crippled the model's reasoning)
4. Weak heuristic (depth-only, non-informative)
5. No verification (hallucinations accepted as goals)

I fixed all five: K=3 separate API calls, self-consistency voting, CoT-hybrid prompts, entity-coverage heuristic, context grounding. After fixes: A* outperformed CoT on multi-hop questions.

**Result:** The paper was submitted to TMLR. The learning: when a principled approach gives bad results, the implementation is usually wrong — not the principle. Debug systematically before abandoning the approach.

---

### Q210: Where do you see yourself in 3-5 years?

**Situation:** Showing vision and ambition.

**Task:** Connect past work to future direction.

**Action:** I want to work at the frontier of LLM reasoning and agent systems — either at a research lab (Anthropic, DeepMind, OpenAI) pushing inference-time compute further, or as a founding engineer at an AI agent company building production-grade autonomous systems. My thesis proved that structured search over reasoning traces works; my Mercedes experience proved I can ship production AI. I want to combine both: build systems that are both rigorous and real.

**Result:** Near-term: AI Engineer at a company pushing agents/reasoning. Long-term: leading an AI team building systems that make consequential decisions autonomously with formal correctness guarantees.
