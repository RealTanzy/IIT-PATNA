# Chapter 6: RAG (Retrieval-Augmented Generation) — Complete Guide

## 6.1 What Is RAG?

Retrieval-Augmented Generation (RAG) is a pattern where an LLM's response is grounded in retrieved documents rather than relying solely on its parametric knowledge. The pipeline:

$$\text{Query} \xrightarrow{\text{retrieve}} \text{Relevant Chunks} \xrightarrow{\text{augment}} \text{LLM Prompt} \xrightarrow{\text{generate}} \text{Answer}$$

**Why RAG instead of fine-tuning?**
- Knowledge updates without retraining (just update the document store)
- Attributable answers (can cite sources)
- Reduced hallucination (model answers from evidence, not memory)
- Works with any LLM (no model access needed for fine-tuning)

**When RAG fails:**
- Questions requiring reasoning across 10+ documents simultaneously
- Real-time data that changes faster than re-indexing
- Queries that need information not in any document

---

## 6.2 The RAG Pipeline: Step by Step

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Processing │ ← Intent classification, expansion, reformulation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │ ← BM25 (sparse) + ANN (dense) in parallel
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Fusion (RRF)  │ ← Merge ranked lists from multiple retrievers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reranking     │ ← Cross-encoder scores each (query, doc) pair
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Compression    │ ← Dedup, truncate, enforce token budget
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation    │ ← LLM with system prompt + context + query
└─────────────────┘
```

---

## 6.3 Sparse Retrieval: BM25

### The Okapi BM25 Formula

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

Where:
- $f(t, d)$ = frequency of term $t$ in document $d$
- $|d|$ = length of document $d$ (in tokens)
- $\text{avgdl}$ = average document length in the corpus
- $k_1 = 1.2$ (term frequency saturation parameter)
- $b = 0.75$ (length normalization parameter)
- $\text{IDF}(t) = \log\frac{N - n(t) + 0.5}{n(t) + 0.5}$ where $N$ = total docs, $n(t)$ = docs containing $t$

**Intuition:**
- Rare terms (high IDF) contribute more than common terms
- Term frequency has diminishing returns (saturates at $k_1$)
- Long documents are penalized (they match more terms by chance)

**Strengths:** Exact keyword matching (ticket IDs, team names, product codes)  
**Weaknesses:** No semantic understanding ("automobile" ≠ "car")

---

## 6.4 Dense Retrieval: Embeddings & ANN

### Bi-Encoder Architecture

A bi-encoder maps queries and documents into the same vector space independently:

```python
query_vec = encode(query)      # [384] dimensional vector
doc_vec = encode(document)     # [384] dimensional vector
score = cosine_similarity(query_vec, doc_vec)
```

**Model used:** `all-MiniLM-L6-v2` (384 dimensions, fast, good quality)

**Cosine similarity:**
$$\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{|\mathbf{q}| \cdot |\mathbf{d}|}$$

### Approximate Nearest Neighbor (ANN) Search

For millions of documents, exact nearest-neighbor is too slow ($O(n)$). ANN algorithms trade small accuracy loss for massive speed:

| Algorithm | Idea | Used In |
|-----------|------|---------|
| **HNSW** | Navigable small-world graph with skip-list layers | FAISS, LanceDB |
| **IVF** | Cluster centroids → only search nearest clusters | FAISS |
| **Flat** | Brute-force (exact, slow) | Small collections |

**HNSW (Hierarchical Navigable Small World):**
- Build a multi-layer graph where each layer has fewer nodes
- Search starts at the top (sparse) layer, descends to the bottom (dense) layer
- Time complexity: $O(\log n)$ queries
- Space: $O(n \cdot M)$ where $M$ = max connections per node

---

## 6.5 Cross-Encoder Reranking

### Why Rerank?

Bi-encoders are fast but approximate. They encode query and document *independently* — no cross-attention. A cross-encoder sees both simultaneously:

```python
# Bi-encoder (fast, approximate):
score = cosine(encode(query), encode(doc))

# Cross-encoder (slow, precise):
score = cross_encoder.predict([(query, doc)])[0]
```

**Cross-encoder input:** `[CLS] query [SEP] document [SEP]` → single relevance score

**Model used:** `ms-marco-MiniLM-L-6-v2` (trained on MS MARCO passage ranking)

### Two-Stage Retrieval

```
Stage 1: Bi-encoder retrieves top-100 candidates (fast, recall-focused)
Stage 2: Cross-encoder reranks top-100 → top-5 (slow, precision-focused)
```

**Why not just use cross-encoder?** It's $O(n)$ — scores every (query, document) pair. For 100K documents, that's 100K forward passes. With bi-encoder first, we only rerank ~10-100 candidates.

---

## 6.6 Hybrid Retrieval & Reciprocal Rank Fusion (RRF)

### Why Hybrid?

BM25 catches exact matches (ticket IDs, team names). Dense retrieval catches semantic matches ("performance issues" ≈ "slow response times"). Neither alone is sufficient.

### Reciprocal Rank Fusion (RRF)

Given multiple ranked lists, RRF merges them with the formula:

$$\text{RRF}(d) = \sum_{r \in R} \frac{w_r}{k + \text{rank}_r(d)}$$

Where:
- $R$ = set of rankers (BM25, ANN)
- $k = 60$ (smoothing constant — prevents rank 1 from dominating)
- $w_r$ = weight for ranker $r$ (default: ANN=0.6, BM25=0.4)
- $\text{rank}_r(d)$ = rank of document $d$ in ranker $r$'s list (1-indexed)

**Example:**
- Document X: rank 3 in ANN, rank 7 in BM25
- RRF(X) = 0.6/(60+3) + 0.4/(60+7) = 0.6/63 + 0.4/67 = 0.00952 + 0.00597 = 0.01549

**Why k=60?** From the original paper (Cormack et al., 2009). It ensures that all ranks contribute meaningfully — rank 1 isn't overwhelmingly better than rank 2.

---

## 6.7 Vector Databases

| Database | Type | Key Feature | Used In |
|----------|------|-------------|---------|
| **LanceDB** | Embedded (local) | Hybrid FTS+ANN in one query, PyArrow schema | SSV RAG (production) |
| **ChromaDB** | Embedded (local) | Simple API, per-collection persistence | SPOT CHECK V2 |
| **FAISS** | Library (in-memory) | Fastest ANN, GPU support, many index types | SSV RAG (prototype) |
| **Pinecone** | Cloud-managed | Serverless, metadata filtering | Not used (corporate data can't leave premises) |
| **Qdrant** | Self-hosted | Rich filtering, sparse+dense | SSV RAG (explored) |

### LanceDB (Used in SSV RAG Production)

```python
import lancedb

db = lancedb.connect("storage/vector_db")
table = db.create_table("knowledge_base", data=records, schema=schema)

# Hybrid query (ANN + FTS in one call):
results = table.search(query_vec).limit(20).to_pandas()  # ANN
results_fts = table.search(query_text, query_type="fts").limit(20).to_pandas()  # BM25
```

Advantages: disk-persistent, no separate server, FTS index built-in, Apache Arrow columnar format.

---

## 6.8 Chunking Strategies

| Strategy | How | Best For |
|----------|-----|----------|
| **Fixed-size** | Split every N tokens with overlap | General text, no structure |
| **Semantic** | Split at sentence/paragraph boundaries | Natural language documents |
| **Section-aware** | Split at headers/sections, preserve hierarchy | Structured docs (Confluence) |
| **Table-aware** | Each table row = one chunk (with column headers) | Tabular data (Jira, wikis) |
| **Sliding window** | Overlapping windows of size N, step S | When context continuity matters |

### SSV RAG Chunking (Table-Aware)

Confluence weekly updates are HTML tables. Each row represents one team's update for one calendar week. The chunker:

1. Parses HTML → extracts table rows
2. Each row becomes a chunk: `{team}_{CW}_{content}`
3. Column headers injected into every chunk (so each chunk is self-contained)
4. Non-table text: sentence-boundary splitting with 50-token overlap
5. Metadata: team name, calendar week, source page, section path

This ensures retrieval for "What did team EDS do in CW24?" returns exactly the right row.

---

## 6.9 Caching: Semantic, Disk, and API

### Three-Tier Caching Architecture (SSV RAG)

```
Tier 1: Semantic Cache (in-memory, cosine similarity)
    ↓ miss
Tier 2: LLM Disk Cache (diskcache, MD5 prompt hash)
    ↓ miss
Tier 3: Full Pipeline (retrieve → rerank → generate)
```

**Tier 1 — Semantic Cache:**
```python
class SemanticCache:
    threshold = 0.92  # Cosine similarity
    
    def get(self, query):
        query_vec = embed(query)
        for cached_query, cached_answer, cached_vec in self.entries:
            if cosine_sim(query_vec, cached_vec) >= self.threshold:
                return cached_answer  # HIT
        return None  # MISS
```

**Why 0.92?** Lower threshold → more cache hits but risk of returning wrong answers. Higher → fewer hits. 0.92 is the sweet spot where "What bugs does EDS have?" matches "Show me EDS bugs" but NOT "What bugs does BSW have?"

**Tier 2 — LLM Disk Cache:**
```python
cache = diskcache.Cache("storage/llm_cache", size_limit=500_000_000)  # 500MB
key = hashlib.md5(prompt.encode()).hexdigest()
if key in cache:
    return cache[key]
```

TTL: 30 minutes (data changes frequently).

**Tier 3 — Connection Pooling:**
```python
session = requests.Session()
adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3)
session.mount("https://", adapter)
```

Reuses TCP connections → eliminates TLS handshake per request.

---

## 6.10 Latency Optimization: 22s → 4s

The SSV RAG chatbot had a latency problem: 22 seconds per query. Here's how each fix contributed:

| # | Problem | Fix | Savings |
|---|---------|-----|---------|
| 1 | Models re-loaded per request | `@st.cache_resource` singletons | ~5s |
| 2 | New HTTP connection per Jira call | Connection pool (10 persistent) | ~1-2s |
| 3 | Sequential SWF + MCP queries | `ThreadPoolExecutor` parallel | ~1s |
| 4 | 16KB context sent to LLM | Context compression (15K chars max) | ~8-12s LLM time |
| 5 | No caching | Semantic cache + disk cache | Variable (repeat queries → 0s) |

**Additional micro-optimizations:**
- Pre-warm models on app startup (dummy inference fills CUDA cache)
- Pre-compute 6 quick-action queries in background thread
- Cap reranker input to 10 candidates (was reranking all 50)
- Intent-based `max_output_tokens` (count queries → 512 tokens instead of 8192)

**Net result:** 22s → 4s median latency (5.5× improvement).

---

## 6.11 Query Processing: Intent Classification

Not all queries need the same pipeline:

```python
class Intent(Enum):
    META = "meta"         # "What can you do?" → no retrieval needed
    SEARCH = "search"     # "Show me EDS bugs" → standard RAG
    RANGE = "range"       # "Compare CW20 to CW24" → multi-week retrieval
    COMPARISON = "compare" # "Which team has most bugs?" → aggregate
    IMAGE = "image"       # User uploaded an image → multimodal
    FOLLOW_UP = "follow"  # References previous answer → use context
```

Each intent routes to a different pipeline path:
- META: Return canned response (no LLM call)
- SEARCH: Full RAG pipeline
- RANGE: Retrieve per-CW, merge, then generate
- COMPARISON: Aggregate from structured data (no retrieval)
- IMAGE: Pass to vision API
- FOLLOW_UP: Append to conversation context, re-retrieve if needed

---

## 6.12 Multi-LLM Backend with Fallback

The system supports multiple LLM backends (Claude Opus + Gemini 2.5 Pro) with automatic failover:

```python
def generate(prompt, context):
    try:
        return call_opus(prompt, context)  # Primary
    except (TimeoutError, RateLimitError, ServerError):
        return call_gemini(prompt, context)  # Fallback
```

**Claude Opus (primary):**
- Via MBRDI Nexus → AWS Bedrock
- Streaming: Amazon EventStream binary protocol
- Token limit management: 200K context window

**Gemini 2.5 Pro (secondary):**
- Via GenAI Nexus internal gateway
- SDK: `google.genai.Client`
- Streaming: `generate_content_stream()`

---

## 6.13 Evaluation: The 100-Query Benchmark

To measure quality objectively, we built a 100-query evaluation framework:

| Category | Queries | What It Tests |
|----------|---------|---------------|
| Jira SWF (Q1-Q25) | On-premise issue tracking | Ticket lookup, status counts |
| Jira MCP (Q26-Q50) | Cloud issue tracking | Cloud API integration |
| Confluence (Q51-Q75) | Wiki weekly updates | Table extraction, CW filtering |
| Mixed + Guardrail (Q76-Q100) | Cross-source + safety | OOD rejection, multi-source merge |

**Metrics measured:**
- **Answer rate:** % of queries that produce a substantive answer (not "I don't know")
- **Guardrail compliance:** % of OOD queries correctly deflected
- **Context hit rate:** % of times retrieved chunks contain the answer
- **Latency:** P50, P95, P99 response times

**Results:**
- Answer rate: **77%** (77/100 queries answered)
- Guardrail compliance: **100%** (10/10 OOD queries rejected)
- Context hit rate: 80% (Confluence), 80% (SWF Jira)

---

## 6.14 Case Study: SSV RAG Chatbot — Full Architecture

### System Overview

A production RAG chatbot serving the Mercedes-AMG System Software Validation team. Answers questions grounded in 3 enterprise data sources.

### Data Sources
1. **Jira SWF** (on-premise): PAT auth, REST v2, 200-item pagination
2. **Jira MCP** (cloud): Basic auth, REST v3, nextPageToken pagination
3. **Confluence** (cloud): CQL search + recursive child page crawl

### Pipeline (LangGraph StateGraph)

```
classify_intent → [META: shortcircuit]
                → [FOLLOW_UP: shortcircuit]  
                → expand → retrieve → generate → END
```

With special paths for RANGE (per-CW retrieval) and IMAGE (multimodal).

### Knowledge Base Sync

Background daemon thread (`KBSyncManager`):
- Runs every 30 minutes
- Delta sync: only re-embeds documents where `updated_at` changed
- Thread-safe with lock
- Manual trigger from UI

### Key Design Decisions

1. **LanceDB over FAISS:** Needed disk persistence (FAISS is in-memory), hybrid FTS+ANN in one store, and PyArrow schema enforcement.

2. **Cross-encoder capped at 10 candidates:** Reranking 50 candidates took 3s; reranking 10 took 0.4s with negligible quality loss (top-5 results were the same 95% of the time).

3. **Semantic cache at 0.92 threshold:** Tested 0.85 (too aggressive — returned wrong answers for similar queries about different teams), 0.95 (too conservative — rarely hit), 0.92 was the sweet spot.

4. **Dual LLM backend:** Corporate policy required no single-vendor lock-in. Hot-swappable via config without code changes.

---

## 6.15 Summary & Interview Tips

**If asked "Design a RAG system":**
1. Start with the pipeline: ingest → chunk → embed → index → retrieve → rerank → generate
2. Mention hybrid retrieval (BM25 for lexical, ANN for semantic)
3. Explain why cross-encoder reranking matters (bi-encoder misses)
4. Discuss chunking strategy (depends on document structure)
5. Address caching (semantic + disk) and latency
6. Mention evaluation (how do you know it works?)

**If asked "How did you optimize latency?":**
Walk through the 5 fixes in order. Each has a clear before/after measurement.

**If asked "BM25 vs dense retrieval?":**
- BM25: exact keywords, ticket IDs, names. No semantic understanding.
- Dense: semantic similarity. Misses exact strings.
- Answer: use BOTH, merge with RRF. Neither alone is sufficient.

**If asked "How do you evaluate a RAG system?":**
- Build a query benchmark (100+ queries, categorized by source/difficulty)
- Measure: answer rate, context hit rate, guardrail compliance, latency
- Don't just measure "accuracy" — measure what % of questions get ANY answer
