# Chapter 13: Python Engineering for AI

## 13.1 asyncio

Modern AI systems spend most of their time waiting — waiting for LLM APIs, waiting for database queries, waiting for vector search results. asyncio is Python's answer to this problem: a single-threaded event loop that multiplexes I/O without the overhead of thread context switching.

### The Event Loop Model

The event loop runs on a single thread and maintains a queue of tasks. When a coroutine hits an `await` (an I/O boundary), it yields control back to the loop, which picks up the next ready task. This is cooperative multitasking — no preemption, no race conditions on shared state.

```python
import asyncio

async def call_llm(prompt: str) -> str:
    """Non-blocking LLM call — yields while waiting for response."""
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_URL, json={"prompt": prompt}) as resp:
            return await resp.json()

async def process_queries(queries: list[str]) -> list[str]:
    """Run multiple LLM calls concurrently, not sequentially."""
    tasks = [call_llm(q) for q in queries]
    return await asyncio.gather(*tasks)
```

### asyncio.gather() — Concurrent Execution

`gather()` schedules multiple coroutines and waits for all of them to complete. This is the bread and butter of concurrent I/O in AI pipelines:

```python
# Sequential: 5 API calls × 2s each = 10s total
# Concurrent: 5 API calls via gather = ~2s total (limited by slowest)
results = await asyncio.gather(
    fetch_embeddings(chunks),
    search_bm25(query),
    search_ann(query),
    get_user_context(user_id),
    log_request(request_id),
)
```

### Semaphores — Rate Limiting

Most APIs have rate limits. A semaphore prevents you from overwhelming them:

```python
sem = asyncio.Semaphore(5)  # Max 5 concurrent requests

async def rate_limited_call(prompt: str) -> str:
    async with sem:
        return await call_llm(prompt)
```

### When NOT to Use asyncio

asyncio shines for I/O-bound work (network calls, file reads, database queries). For CPU-bound work (embedding computation, data transformation, numerical processing), use `ProcessPoolExecutor` instead. The GIL means CPU-bound coroutines will block the entire event loop.

---

## 13.2 Concurrency Patterns

### ThreadPoolExecutor — I/O Parallelism

When you have synchronous I/O libraries (most database drivers, some SDKs), wrap them in a thread pool:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def search_bm25(query: str) -> list[dict]:
    """Synchronous BM25 search — blocks thread, not process."""
    return index.search(query, top_k=50)

def search_ann(query: str) -> list[dict]:
    """Synchronous ANN search — blocks thread, not process."""
    vec = model.encode(query)
    return vector_store.query(vec, top_k=50)

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(search_bm25, query): "bm25",
        executor.submit(search_ann, query): "ann",
    }
    results = {}
    for future in as_completed(futures):
        results[futures[future]] = future.result()
```

### ProcessPoolExecutor — CPU Parallelism

For CPU-bound work, processes bypass the GIL entirely:

```python
from concurrent.futures import ProcessPoolExecutor

def embed_batch(texts: list[str]) -> np.ndarray:
    """CPU-intensive embedding — runs in separate process."""
    return model.encode(texts, batch_size=32)

chunks = [texts[i:i+100] for i in range(0, len(texts), 100)]
with ProcessPoolExecutor(max_workers=4) as executor:
    embeddings = list(executor.map(embed_batch, chunks))
```

### The GIL Explained

The Global Interpreter Lock prevents multiple threads from executing Python bytecode simultaneously. However, it is released during:
- I/O operations (network, disk)
- C extension calls (numpy, PyTorch)
- Explicit releases (time.sleep)

This means threads ARE useful for I/O-bound code and C-extension-heavy code. They are NOT useful for pure Python computation.

### Real Example: SSV RAG Hybrid Search

```python
# BM25 and ANN run in parallel threads because both release the GIL
# (BM25 uses a C library, ANN uses FAISS/LanceDB C++ backend)
with ThreadPoolExecutor(max_workers=2) as pool:
    bm25_future = pool.submit(bm25_search, query_tokens)
    ann_future = pool.submit(ann_search, query_embedding)
    bm25_results = bm25_future.result()
    ann_results = ann_future.result()
# Fuse with RRF — pure Python, runs on main thread
fused = reciprocal_rank_fusion(bm25_results, ann_results, k=60)
```

---

## 13.3 API Design with FastAPI

### Endpoint Design

FastAPI combines type hints with automatic OpenAPI documentation. The key principles for AI APIs:

```python
from fastapi import FastAPI, BackgroundTasks, Depends
from pydantic import BaseModel, Field

app = FastAPI(title="RAG Service", version="2.0")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, str] | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float = Field(ge=0.0, le=1.0)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    ...
```

### Dependency Injection

Shared resources (model instances, DB connections) are injected via `Depends`:

```python
def get_retriever() -> HybridRetriever:
    return app.state.retriever  # Loaded once at startup

@app.post("/query")
async def query(request: QueryRequest, retriever=Depends(get_retriever)):
    results = await retriever.search(request.query, top_k=request.top_k)
    ...
```

### Streaming Responses

For LLM output, stream tokens as they arrive:

```python
from fastapi.responses import StreamingResponse

@app.post("/stream")
async def stream_query(request: QueryRequest):
    async def generate():
        async for token in llm.stream(request.query):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 13.4 Design Patterns for AI Systems

### Singleton — Load Once, Share Everywhere

ML models are expensive to load. The singleton pattern ensures they load exactly once:

```python
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# FastAPI equivalent
@app.on_event("startup")
async def startup():
    app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
```

### Adapter — Unified Interface for Multiple Formats

Different document formats need different parsers, but the pipeline expects a single interface:

```python
class DocumentAdapter(ABC):
    @abstractmethod
    def extract_text(self, file_path: Path) -> list[TextChunk]: ...

class PDFAdapter(DocumentAdapter):
    def extract_text(self, file_path):
        doc = fitz.open(file_path)
        return [TextChunk(page.get_text()) for page in doc]

class DOCXAdapter(DocumentAdapter):
    def extract_text(self, file_path):
        doc = Document(file_path)
        return [TextChunk(para.text) for para in doc.paragraphs]
```

### Registry — Plugin Auto-Discovery

Decorators register handlers dynamically:

```python
INTENT_REGISTRY = {}

def register(priority: int):
    def decorator(cls):
        INTENT_REGISTRY[cls.intent_name] = (priority, cls)
        return cls
    return decorator

@register(priority=1)
class TicketLookupHandler:
    intent_name = "ticket_lookup"
    ...
```

### Strategy — Swappable Algorithms

Same interface, different implementations:

```python
class RerankStrategy(Protocol):
    def rerank(self, query: str, docs: list[dict]) -> list[dict]: ...

class CrossEncoderRerank:
    def rerank(self, query, docs):
        scores = self.model.predict([(query, d["text"]) for d in docs])
        return sorted(zip(docs, scores), key=lambda x: -x[1])

class LLMRerank:
    def rerank(self, query, docs):
        # Use LLM to judge relevance — slower but more accurate
        ...
```

### Mediator — Agent Coordination

SPOT CHECK's `AgentCoordinator` orchestrates multiple specialized agents without them knowing about each other:

```python
class AgentCoordinator:
    def __init__(self):
        self.agents = {"parser": ParserAgent(), "checker": CheckerAgent(), ...}
    
    async def process(self, document):
        parsed = await self.agents["parser"].run(document)
        checks = await asyncio.gather(*[
            self.agents["checker"].check(item) for item in parsed.items
        ])
        return self.agents["reporter"].summarize(checks)
```

---

## 13.5 Error Handling & Resilience

### Retry with Exponential Backoff

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def call_llm(prompt: str) -> str:
    response = await client.invoke(model_id=MODEL, body=prompt)
    return response["completion"]
```

### Circuit Breaker

After N consecutive failures, stop calling the service entirely for a cooldown period:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown=60):
        self.failures = 0
        self.last_failure = 0
        self.threshold = failure_threshold
        self.cooldown = cooldown
    
    def call(self, fn, *args):
        if self.failures >= self.threshold:
            if time.time() - self.last_failure < self.cooldown:
                raise CircuitOpenError("Service unavailable")
            self.failures = 0  # Try again after cooldown
        try:
            result = fn(*args)
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            self.last_failure = time.time()
            raise
```

### Fallback Strategy

```python
async def generate_answer(query: str, context: str) -> str:
    try:
        return await call_primary_model(query, context)  # Claude via Bedrock
    except (TimeoutError, ServiceUnavailableError):
        return await call_fallback_model(query, context)  # Local smaller model
```

---

## 13.6 Caching Patterns

### LRU Cache — In-Memory

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def get_embedding(text: str) -> np.ndarray:
    return model.encode(text)
```

### Semantic Cache

Instead of exact-match caching, find queries that are semantically similar:

```python
class SemanticCache:
    def __init__(self, threshold=0.92):
        self.cache = []  # (embedding, response) pairs
        self.threshold = threshold
    
    def get(self, query_embedding):
        for cached_emb, response in self.cache:
            if cosine_similarity(query_embedding, cached_emb) > self.threshold:
                return response
        return None
```

### TTL-Based — Different Lifetimes Per Intent

```python
TTL_CONFIG = {
    "ticket_status": 300,      # 5 min — tickets change frequently
    "documentation": 3600,     # 1 hour — docs are stable
    "team_info": 86400,        # 1 day — org structure rarely changes
}
```

---

## 13.7 Logging & Observability

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

logger.info("query_processed",
    request_id=request_id,
    query_length=len(query),
    retrieval_time_ms=retrieval_ms,
    num_chunks_retrieved=len(chunks),
    model_used="claude-3-sonnet",
    total_tokens=usage.total_tokens,
)
```

### Context Propagation

Every request gets a unique ID that flows through the entire pipeline:

```python
@app.middleware("http")
async def add_request_id(request, call_next):
    request_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

This allows tracing a single user query through retrieval, reranking, generation, and response — critical for debugging production issues in multi-component AI systems.
