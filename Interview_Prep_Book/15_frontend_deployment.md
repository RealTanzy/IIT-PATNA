# Chapter 15: Frontend & Deployment

## 15.1 Streamlit for AI Applications

Streamlit is the fastest path from "working Python script" to "internal tool your team can use." For AI applications — where the core value is the model, not the UI — Streamlit eliminates the frontend/backend divide entirely. You write Python; you get a web app.

### Multi-Page Apps

```python
# pages/1_Chat.py
import streamlit as st

st.set_page_config(page_title="RAG Chat", layout="wide")

# Sidebar navigation happens automatically via pages/ directory structure
# pages/1_Chat.py
# pages/2_Admin.py
# pages/3_Analytics.py

st.title("RAG Chat Interface")
```

### Session State — Persistence Across Reruns

Streamlit reruns the entire script on every interaction. Session state is how you persist data:

```python
# Initialize state on first run
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever()

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### Streaming — Token-by-Token Output

```python
def generate_response(query: str):
    """Generator that yields tokens for st.write_stream()."""
    context = retriever.search(query)
    for chunk in llm.stream(query=query, context=context):
        yield chunk.text

# In the chat interface:
with st.chat_message("assistant"):
    response = st.write_stream(generate_response(prompt))
```

### Caching — Expensive Operations Run Once

```python
@st.cache_resource  # Singleton — shared across all sessions
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def connect_vector_store():
    return lancedb.connect("./vector_store")

@st.cache_data(ttl=3600)  # Data cache — serialized, per-input, expires after 1 hour
def get_document_stats():
    return {"total_docs": store.count(), "last_sync": store.last_modified()}
```

### File Upload

```python
uploaded = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "xlsx", "csv"],
    accept_multiple_files=True,
)
if uploaded:
    for file in uploaded:
        chunks = ingest_document(file)
        st.success(f"Ingested {file.name}: {len(chunks)} chunks")
```

### Limitations

- **Single-user concurrency**: Each user gets their own script execution, but heavy computation blocks that user's session
- **No websockets**: Cannot push updates from server to client
- **Rerun model**: Every widget interaction reruns the entire script (mitigated by caching)
- **No fine-grained layout control**: Limited compared to React/Vue

Despite these limitations, Streamlit is the right choice for 80% of internal AI tools where development speed matters more than UI polish.

---

## 15.2 PySide6/Qt for Desktop Apps

When you need offline operation, complex interactivity, or native performance, PySide6 (Qt for Python) is the answer. Use cases: lab instruments, signal processing tools, annotation interfaces.

### Signals & Slots — Event-Driven Programming

```python
from PySide6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import Signal, Slot, QThread

class AnalysisWorker(QThread):
    """Background thread for CPU-intensive analysis."""
    progress = Signal(int)       # Emit progress updates
    finished = Signal(object)    # Emit results when done
    
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def run(self):
        for i, batch in enumerate(self.data):
            result = heavy_computation(batch)
            self.progress.emit(int(i / len(self.data) * 100))
        self.finished.emit(result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.button = QPushButton("Run Analysis")
        self.button.clicked.connect(self.start_analysis)
    
    @Slot()
    def start_analysis(self):
        self.worker = AnalysisWorker(self.data)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.show_results)
        self.worker.start()  # Non-blocking — UI stays responsive
```

### Real-Time Plotting with pyqtgraph

```python
import pyqtgraph as pg

class SignalViewer(QMainWindow):
    """Real-time signal visualization (iFAST use case)."""
    
    def __init__(self):
        super().__init__()
        self.plot_widget = pg.PlotWidget()
        self.curve = self.plot_widget.plot(pen='g')
        
        # Timer for real-time updates (60 FPS)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16)  # ~60 Hz
    
    def update_plot(self):
        new_data = self.data_source.get_latest()
        self.curve.setData(new_data)
```

---

## 15.3 Docker for AI Systems

AI applications have complex dependencies — CUDA drivers, large model files, specific library versions. Docker makes deployment reproducible.

### Multi-Stage Build

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime (smaller image)
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY config/ ./config/

# Don't run as root
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Model Caching with Volumes

```yaml
# docker-compose.yml
services:
  rag-app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - model-cache:/app/models        # Persist model weights
      - ./vector_store:/app/vector_store  # Persist embeddings
    env_file:
      - .env  # Never bake secrets into the image
    
  lancedb:
    image: lancedb/lancedb:latest
    volumes:
      - lance-data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  model-cache:
  lance-data:
```

### GPU Support

```dockerfile
# For models requiring GPU inference
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.11 python3-pip
# The nvidia-container-toolkit handles GPU passthrough at runtime
```

```yaml
# docker-compose.yml with GPU
services:
  inference:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Environment Variables — Never Bake Secrets

```dockerfile
# WRONG: secrets baked into image (visible in docker history)
ENV API_KEY=sk-1234567890

# RIGHT: secrets injected at runtime
# docker run --env-file .env myapp
# or via orchestrator (ECS task definition, K8s secrets)
```

---

## 15.4 AWS Bedrock

AWS Bedrock provides managed LLM inference without managing GPU infrastructure. You call an API; AWS handles scaling, availability, and hardware.

### Basic Invocation

```python
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

def invoke_claude(prompt: str, max_tokens: int = 1024) -> str:
    response = client.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }),
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]
```

### Streaming — Token-by-Token

```python
def stream_claude(prompt: str):
    """Yield tokens as they arrive — critical for UX."""
    response = client.invoke_model_with_response_stream(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
        }),
    )
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            yield chunk["delta"]["text"]
```

### Authentication

In corporate environments, Bedrock access typically goes through:
- IAM roles attached to EC2/ECS tasks (no keys in code)
- Corporate proxy with pre-configured credentials
- Cross-account assume-role for shared model access

### Benefits Over Self-Hosted

- No GPU procurement or management
- Pay-per-token (no idle cost)
- Auto-scaling to handle traffic spikes
- Multiple model providers (Claude, Llama, Mistral) behind one API
- Built-in guardrails and content filtering

---

## 15.5 Monitoring & Evaluation

### Latency Tracking

Measure each component independently to find bottlenecks:

```python
import time

class LatencyTracker:
    def __init__(self):
        self.timings = {}
    
    def track(self, component: str):
        """Context manager for timing components."""
        class Timer:
            def __enter__(timer_self):
                timer_self.start = time.perf_counter()
                return timer_self
            def __exit__(timer_self, *args):
                elapsed_ms = (time.perf_counter() - timer_self.start) * 1000
                self.timings[component] = elapsed_ms
        return Timer()

# Usage:
tracker = LatencyTracker()
with tracker.track("retrieval"):
    chunks = retriever.search(query)
with tracker.track("reranking"):
    chunks = reranker.rerank(query, chunks)
with tracker.track("generation"):
    answer = llm.generate(query, chunks)

# Log: {"retrieval": 120, "reranking": 85, "generation": 2100}
logger.info("request_complete", **tracker.timings)
```

### Error Rate Monitoring

```python
from collections import Counter

error_counter = Counter()

# Track by type
error_counter["timeout"] += 1
error_counter["rate_limit"] += 1
error_counter["generation_failure"] += 1
error_counter["empty_retrieval"] += 1

# Alert if error rate exceeds threshold
total_requests = sum(error_counter.values()) + success_count
error_rate = sum(error_counter.values()) / total_requests
if error_rate > 0.05:  # >5% error rate
    alert("Error rate elevated", error_counter)
```

### Quality Metrics — Evaluation Benchmark

```python
class QualityBenchmark:
    """100-query evaluation suite run nightly."""
    
    def __init__(self, test_cases: list[dict]):
        self.test_cases = test_cases  # [{query, expected_answer, expected_sources}]
    
    def run(self) -> dict:
        results = {"answer_rate": 0, "retrieval_recall": 0, "faithfulness": 0}
        
        for case in self.test_cases:
            response = rag_pipeline.query(case["query"])
            
            # Answer rate: did we produce an answer (not "I don't know")?
            if not response.is_abstention:
                results["answer_rate"] += 1
            
            # Retrieval recall: did we find the right sources?
            retrieved_ids = {c.source_id for c in response.chunks}
            expected_ids = set(case["expected_sources"])
            results["retrieval_recall"] += len(retrieved_ids & expected_ids) / len(expected_ids)
            
            # Faithfulness: is the answer supported by retrieved context?
            results["faithfulness"] += self._judge_faithfulness(response)
        
        # Normalize
        n = len(self.test_cases)
        return {k: v / n for k, v in results.items()}
```

---

## 15.6 CI/CD for AI Applications

### The Testing Pyramid for AI

```
         /\
        /  \        Evaluation (nightly)
       / $$ \       - 100-query benchmark
      /------\      - LLM-as-judge scoring
     /        \     - Expensive, slow (~30 min)
    / Integr.  \    
   /   Tests    \   Integration (per-PR)
  /   (medium)   \  - End-to-end pipeline test
 /________________\  - Mock external APIs
/    Unit Tests    \ Unit (per-commit)
/    (fast, free)   \ - Pure functions, chunking logic
/____________________\ - No external dependencies
```

### Pre-Commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff          # Lint
      - id: ruff-format   # Format
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy          # Type check
        additional_dependencies: [pydantic, fastapi]
```

### CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[test]"
      - run: pytest tests/unit/ -x --tb=short
      - run: pytest tests/integration/ -x --tb=short
  
  # Evaluation runs nightly, not per-PR (too expensive)
  evaluate:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/evaluation/ --benchmark-output=results.json
      - uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: results.json
```

### Blue-Green Deployment

For AI systems, you cannot just deploy and hope. Quality might degrade silently:

```python
class BlueGreenDeployment:
    """Run both versions, compare quality before switching."""
    
    async def shadow_test(self, query: str):
        # Run both versions
        blue_response = await self.blue.query(query)   # Current production
        green_response = await self.green.query(query)  # New version
        
        # Log comparison (don't serve green to users yet)
        log_comparison(query, blue_response, green_response)
        
        # Serve blue (stable) to user
        return blue_response
    
    def evaluate_green(self) -> bool:
        """After N shadow queries, decide if green is safe to promote."""
        comparisons = load_comparisons()
        green_wins = sum(1 for c in comparisons if c.green_better)
        return green_wins / len(comparisons) > 0.95  # Green must be >= 95% as good
```

---

## 15.7 Summary

The deployment landscape for AI applications:

**Start with Streamlit** — 80% of AI applications begin as Streamlit apps. The speed from idea to working prototype is unmatched. When your internal tool serves 5-50 users, Streamlit is sufficient.

**Containerize with Docker** — Once you need reproducibility, shared deployment, or multiple services (app + vector DB + cache), Docker Compose is the natural next step. Multi-stage builds keep images lean.

**Monitor continuously** — AI systems fail silently. A retrieval regression does not throw an error; it just returns worse results. Latency tracking catches performance regressions. Quality benchmarks catch accuracy regressions.

**Evaluation is NOT optional** — The most common mistake in AI application development is shipping without an evaluation framework. By the time you notice quality has degraded (from user complaints), you have no idea which change caused it. Build your 100-query benchmark on day one. Run it nightly. Alert when metrics drop.

The progression for most teams:
1. Local script (proof of concept)
2. Streamlit app (internal tool)
3. FastAPI + Docker (production service)
4. Full CI/CD + monitoring + evaluation (mature system)

Each step adds complexity but also reliability. Move to the next step only when the current one creates friction.
