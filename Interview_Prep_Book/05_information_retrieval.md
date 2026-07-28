# Chapter 5: Information Retrieval & Search

*The theory behind RAG systems — from lexical matching to hybrid retrieval pipelines.*

---

Information Retrieval (IR) is the foundation beneath every RAG system. Before you can generate an answer grounded in evidence, you must first *find* that evidence. This chapter covers the full retrieval stack: classical sparse methods (TF-IDF, BM25), dense neural retrieval (embeddings, sentence-transformers), re-ranking (cross-encoders), approximate nearest neighbor algorithms, hybrid fusion, vector databases, chunking strategies, and evaluation metrics. By the end, you will understand not just how each component works, but *when* to reach for which tool — the judgment calls that separate production systems from toy demos.

---

## 5.1 The Retrieval Problem

The core problem is deceptively simple: given a user query *q*, return the most relevant documents from a corpus *C* of *N* documents.

In practice, "relevant" is ambiguous. A user searching for "how to fix memory leaks in Python" might benefit from a document titled "Debugging Memory Issues in CPython" even though the exact phrase "memory leaks" never appears. Conversely, a user searching for ticket "JIRA-4821" needs an exact string match — semantic understanding would only hurt.

This tension drives two fundamental tradeoffs:

**Precision vs Recall.** Precision measures what fraction of your returned results are actually relevant. Recall measures what fraction of all relevant documents you successfully retrieved. In RAG, recall matters more than precision at the retrieval stage, because the LLM can ignore irrelevant context — but it cannot hallucinate missing context correctly.

**Lexical matching vs Semantic matching.** Lexical methods (BM25) match on exact tokens. Semantic methods (embeddings) match on meaning. Neither alone is sufficient. The best production systems use both.

---

## 5.2 TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is the simplest effective retrieval algorithm and the conceptual ancestor of BM25.

**Term Frequency (TF):** How often term *t* appears in document *d*. A document mentioning "retrieval" ten times is probably more about retrieval than one mentioning it once.

**Inverse Document Frequency (IDF):** How rare term *t* is across the entire corpus. The word "the" appears in every document — it carries no discriminative signal. The word "HNSW" appears in very few documents — it is highly informative.

**Formula:**

```
TF-IDF(t, d) = TF(t, d) * IDF(t)

where:
  TF(t, d) = count of term t in document d
  IDF(t)   = log(N / df(t))
  N        = total number of documents in corpus
  df(t)    = number of documents containing term t
```

**Limitations:** TF-IDF has no notion of semantics. "Car" and "automobile" are completely unrelated in its world. It performs exact string matching only, with no understanding of synonyms, paraphrases, or intent. It also treats term frequency as linear — a term appearing 100 times is scored 100x higher than one appearing once, which over-rewards verbose documents.

---

## 5.3 BM25 (Okapi BM25)

BM25 is the refinement of TF-IDF that has dominated sparse retrieval for three decades. It addresses TF-IDF's two critical flaws: unbounded term frequency and no length normalization.

**The Full Formula:**

```
BM25(q, d) = SUM over t in q:
  IDF(t) * (TF(t,d) * (k1 + 1)) / (TF(t,d) + k1 * (1 - b + b * (|d| / avgdl)))

where:
  IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5))
  k1     = 1.2 (term frequency saturation parameter)
  b      = 0.75 (length normalization parameter)
  |d|    = length of document d (in tokens)
  avgdl  = average document length across corpus
  n(t)   = number of documents containing term t
  N      = total number of documents
```

**Term Frequency Saturation (k1).** This is *the* key insight of BM25. As TF grows, the score asymptotically approaches (k1 + 1) rather than growing without bound. A document mentioning "Python" 50 times is not 50x more relevant than one mentioning it once — it is only slightly more relevant after a point. With k1=1.2, the marginal value of additional occurrences drops rapidly after the first few.

**Length Normalization (b).** Longer documents naturally contain more term occurrences. The parameter *b* penalizes documents proportional to how much longer they are than average. With b=0.75, a document twice the average length must have proportionally more occurrences to score the same. Setting b=0 disables length normalization; b=1 fully normalizes.

**Why BM25 > TF-IDF:** The saturation curve prevents long, repetitive documents from dominating results. A 10,000-word document repeating "machine learning" hundreds of times will not outscore a concise, focused 500-word explanation.

**When BM25 Excels:**
- Exact matches: ticket IDs (JIRA-1234), error codes (E_NOENT), UUIDs
- Proper nouns: "ChromaDB", "Kubernetes", specific person names
- Code symbols: function names, variable names, class names
- Rare terms: highly specific technical jargon

---

## 5.4 Dense Retrieval & Embeddings

Dense retrieval maps both queries and documents into a shared vector space where semantic similarity corresponds to geometric proximity.

**Embedding:** A function *f* that maps a text string to a fixed-size real-valued vector: f("How do I fix memory leaks?") -> [0.12, -0.34, 0.87, ...] (384 or 768 dimensions typically).

**Training via Contrastive Learning:** The model learns from positive pairs (query, relevant-doc) and negative pairs (query, irrelevant-doc). The training objective pushes positive pairs closer together and negative pairs farther apart in vector space.

**Bi-Encoder Architecture:** Query and document are encoded independently by the same model. This is what makes dense retrieval practical — you can pre-encode all documents offline, then at query time you only encode the query and compute similarities against stored vectors.

**Key Models:**
- **all-MiniLM-L6-v2**: 384 dimensions, 22M parameters, fast inference. The workhorse for prototypes and production.
- **BGE (BAAI General Embedding)**: 768/1024 dimensions, strong multilingual support.
- **E5 (EmbEddings from bidirEctional Encoder rEpresentations)**: Microsoft's model, instruction-tuned variants available.

**Cosine Similarity:**

```
cos(q, d) = (q . d) / (||q|| * ||d||)

For unit-normalized vectors: cos(q, d) = q . d  (dot product)
```

**When Dense > Sparse:** Semantic matching ("automobile" matches "car"), paraphrases ("fix memory issues" matches "resolve RAM problems"), intent understanding ("how to speed up Python" matches "performance optimization techniques").

**When Dense Fails:** Exact strings the model has never seen, rare technical terms not in training data, out-of-domain content where the embedding model has no relevant training signal.

---

## 5.5 Sentence-Transformers

Sentence-Transformers are BERT or DistilBERT models fine-tuned specifically for producing meaningful sentence-level embeddings.

**The Problem They Solve:** Raw BERT produces token-level embeddings. Averaging them into a sentence embedding works poorly — BERT was not trained for this. Sentence-Transformers fine-tune on sentence similarity tasks so the [CLS] token or mean pooling actually captures sentence meaning.

**all-MiniLM-L6-v2:**
- 384 dimensions
- 22M parameters (6-layer DistilBERT variant)
- Fast inference (~14K sentences/sec on GPU)
- Trained on over 1 billion sentence pairs
- Training objective: minimize distance between semantically similar sentences, maximize distance between dissimilar ones

**Normalization:** Vectors are typically L2-normalized to unit length. This means cosine similarity reduces to a simple dot product, which is computationally cheaper and enables optimized ANN index operations.

**Use in My Projects:** Both the SSV RAG system and SPOT CHECK use all-MiniLM-L6-v2 for document encoding. It provides the right balance of speed, quality, and memory footprint for production deployment on standard hardware.

---

## 5.6 Cross-Encoders vs Bi-Encoders

**Bi-Encoder:** Encodes query and document independently through separate forward passes. The similarity is computed post-hoc (dot product or cosine). Because documents are encoded offline, query-time cost is O(1) per pre-encoded document — just a vector comparison.

**Cross-Encoder:** Concatenates query and document as a single input: `[CLS] query [SEP] document [SEP]`. The model sees full token-level interaction between query and document tokens through self-attention. This produces far more accurate relevance scores because the model can attend to fine-grained relationships.

**Why Cross-Encoders are More Accurate:** A bi-encoder must compress all document meaning into a single vector *before* seeing the query. A cross-encoder sees both simultaneously, enabling it to notice that "Python memory" in the query aligns specifically with paragraph 3 of the document about "garbage collection in CPython."

**ms-marco-MiniLM-L-6-v2:** A cross-encoder trained on MS MARCO, Microsoft's passage ranking dataset with 500K+ human-labeled query-passage relevance pairs. This is the standard re-ranker for English-language retrieval.

**Two-Stage Retrieval Pipeline:**
1. **Recall stage:** Bi-encoder retrieves top-100 candidates (fast, O(log n) with ANN)
2. **Precision stage:** Cross-encoder re-ranks top-100 to select top-5 (slow but accurate, O(100) forward passes)

**Why Not Just Cross-Encoder?** A cross-encoder requires a forward pass for *every* query-document pair. For a corpus of 100K documents, that is 100K forward passes per query — completely impractical. The two-stage approach gives you cross-encoder accuracy at bi-encoder speed.

---

## 5.7 Approximate Nearest Neighbor (ANN) Algorithms

Exact nearest neighbor search is O(n) — compare the query vector to every stored vector. For millions of vectors, this is too slow. ANN algorithms trade a small accuracy loss for dramatic speed gains.

### HNSW (Hierarchical Navigable Small World)

The dominant ANN algorithm in production systems today.

**Structure:** A multi-layer graph. Top layers are sparse with long-range connections (like highway links). Bottom layers are dense with short-range connections (like local roads). Each layer is a navigable small-world graph.

**Build Process:** Insert nodes one at a time. Each node is assigned a random maximum layer (exponential distribution). At each layer from top to its maximum, connect it to its M nearest neighbors.

**Search Process:** Start at the entry point in the top layer. Greedily move to the nearest neighbor of the query. When no closer neighbor exists in the current layer, descend one layer. Repeat until reaching layer 0, then return the nearest neighbors found.

**Complexity:** O(log n) query time. Near-perfect recall (>95%) at high throughput.

**Key Parameters:**
- **M** (max connections per node): Higher = better recall, more memory. Typical: 16-64.
- **ef_construction** (build-time beam width): Higher = better graph quality, slower build. Typical: 100-200.
- **ef_search** (query-time beam width): Higher = better recall, slower queries. Typical: 50-200.

### IVF (Inverted File Index)

**Concept:** Cluster all vectors into *nlist* partitions using K-means. At query time, find the nearest cluster centroids, then search only those partitions exhaustively.

**nprobe Tradeoff:** Searching more partitions (higher nprobe) gives better recall but slower queries. Typical: nlist=1024, nprobe=16-64.

### Flat (Brute Force)

Compare the query to every single vector. O(n) time, 100% accurate. Use when your corpus is small (<10K vectors) and the simplicity outweighs the cost.

---

## 5.8 Hybrid Retrieval

Neither sparse nor dense retrieval alone catches everything. They have complementary strengths:

**BM25 catches:**
- Ticket IDs: "JIRA-1234" (embedding models have no special representation for this)
- Exact team/product names: "ChromaDB" vs "Chroma" vs "chromadb"
- Code symbols: `get_embedding_vector()` — the embedding model might match "retrieve vector representation" but miss the exact function name
- Error messages: exact string matching is critical

**Dense catches:**
- Semantic similarity: "fix memory leaks" matches "resolve RAM issues"
- Paraphrases: "how to speed up queries" matches "optimizing database performance"
- Intent: "why is my app slow" matches documentation about performance profiling

**Merging Strategies:**
- **Reciprocal Rank Fusion (RRF):** Rank-based fusion, no score calibration needed
- **Linear combination:** `score = alpha * dense_score + (1-alpha) * sparse_score` (requires score normalization)
- **Max score:** Take the maximum score from either system per document

---

## 5.9 Reciprocal Rank Fusion (RRF)

RRF is the standard method for combining results from multiple rankers without requiring score calibration.

**Formula:**

```
RRF_score(d) = SUM over rankers r:
  weight_r / (k + rank_r(d))

where:
  k = 60 (constant from Cormack et al., 2009)
  rank_r(d) = rank of document d in ranker r's results (1-indexed)
  weight_r = weight assigned to ranker r
```

**Why RRF Works:** BM25 scores are unbounded (a perfect match on a rare term can score 20+). Embedding cosine similarities range from [-1, 1] in theory but cluster in [0.3, 0.9] in practice. You cannot meaningfully add these scores — they are on incompatible scales. RRF uses *rank* (ordinal position), not *score* (cardinal value). Rank 1 is always the best result regardless of the underlying scoring function. This provides natural calibration across heterogeneous rankers.

**Why k=60?** The constant *k* controls how much the fusion rewards top-ranked results vs. lower-ranked ones. With k=60, the difference between rank 1 (score 1/61) and rank 2 (score 1/62) is small, but the difference between rank 1 (1/61) and rank 100 (1/160) is large. This prevents a single ranker's top result from dominating when other rankers disagree.

**Practical Weighting:**

```
ann_weight  = 0.6  (dense retrieval usually better overall)
bm25_weight = 0.4  (but BM25 catches critical edge cases)
```

Dense retrieval generally produces better results for natural language queries, hence the higher weight. But BM25's contribution on exact-match cases (IDs, names, code) is irreplaceable, so it retains significant weight.

---

## 5.10 Vector Databases

Choosing the right vector store depends on your deployment constraints.

**ChromaDB:**
- In-memory, per-collection stores
- Simple Python API, minimal configuration
- Good for prototypes, per-session stores, small corpora
- No built-in persistence across restarts (without explicit persist)
- Limited scalability

**LanceDB:**
- Disk-persistent with memory-mapped access
- Native hybrid search: full-text search (FTS) + ANN in one query
- Apache Arrow columnar format — efficient for structured metadata
- No server needed — embedded library
- Excellent for production RAG systems that need persistence without infrastructure overhead

**FAISS (Facebook AI Similarity Search):**
- Fastest option, especially with GPU acceleration
- Supports IVF, HNSW, PQ (Product Quantization) and combinations
- In-memory only — no built-in persistence (must serialize/deserialize manually)
- Best for maximum throughput when you can afford the RAM

**Pinecone:**
- Fully managed cloud service — no infrastructure to maintain
- Serverless scaling, metadata filtering, namespaces
- Data leaves your premises (compliance consideration)
- Pay-per-query pricing model

**Qdrant:**
- Self-hosted or cloud, rich filtering with payload indexes
- Supports both sparse and dense vectors natively
- HNSW-based with configurable quantization
- Good middle ground: production-ready without vendor lock-in

**Decision Framework:**
| Use Case | Choose |
|----------|--------|
| Prototype / per-session | ChromaDB |
| Production / hybrid / disk | LanceDB |
| Maximum speed / GPU | FAISS |
| Cloud-first / no ops | Pinecone |
| Self-hosted / rich filtering | Qdrant |

---

## 5.11 Chunking Strategies

Documents must be chunked before embedding because: (1) embedding models have token limits (typically 512 tokens), (2) LLM context windows are finite, and (3) smaller chunks improve retrieval precision — a 10,000-word document where only one paragraph is relevant will dilute the embedding signal.

**Fixed-Size Chunking:**
Split every N tokens regardless of content boundaries. Simple to implement but breaks mid-sentence or mid-thought. A chunk boundary might split "the model achieves state-of-the-" / "art performance on GLUE" — destroying meaning.

**Semantic Chunking:**
Split at sentence or paragraph boundaries. Preserves complete thoughts. Use NLP sentence tokenizers (spaCy, NLTK) to find natural break points. Slightly variable chunk sizes but much better embedding quality.

**Section-Aware Chunking:**
Split at document headers (H1, H2, H3 in Markdown; `<h1>`-`<h6>` in HTML). Each section becomes a chunk with its header as metadata. Preserves document structure and makes retrieval results more interpretable.

**Table-Aware Chunking:**
For structured data (CSVs, database exports, spreadsheets): each row becomes a chunk with column headers injected. A row like `| JIRA-4821 | Critical | Memory leak in auth service |` becomes the chunk "Ticket: JIRA-4821, Priority: Critical, Description: Memory leak in auth service." This makes tabular data searchable with natural language queries.

**Sliding Window (Overlapping Chunks):**
Include 50-100 tokens of overlap between adjacent chunks. If an important concept spans a chunk boundary, both chunks will contain it. Increases storage by ~20% but prevents information loss at boundaries.

**Size Considerations:**
- **Too small (< 100 tokens):** Insufficient context for meaningful embedding. "The model" alone is meaningless.
- **Sweet spot (200-500 tokens):** Enough context for a coherent idea, small enough for precise retrieval.
- **Too large (> 1000 tokens):** Embedding signal gets diluted by irrelevant content within the chunk.

---

## 5.12 Evaluation Metrics

You cannot improve what you cannot measure. Retrieval evaluation requires ground-truth relevance labels: for each query, which documents are actually relevant?

**Recall@k:**
```
Recall@k = |relevant docs in top-k| / |total relevant docs|
```
"Of all relevant documents, what fraction did we retrieve in the top k?" For RAG, Recall@10 is critical — if the answer is not in your retrieved context, no amount of generation quality can save you.

**Precision@k:**
```
Precision@k = |relevant docs in top-k| / k
```
"Of the k documents we retrieved, what fraction are relevant?" Less critical for RAG (the LLM can ignore noise) but important for user-facing search results.

**MRR (Mean Reciprocal Rank):**
```
MRR = (1/|Q|) * SUM over queries q: 1 / rank_q(first relevant doc)
```
"On average, how high does the first relevant result appear?" MRR=1.0 means the first result is always relevant. MRR=0.5 means the first relevant result is typically at position 2.

**NDCG (Normalized Discounted Cumulative Gain):**
Accounts for graded relevance (not just binary relevant/not-relevant) and penalizes relevant documents appearing lower in the ranking. The "discounted" refers to a logarithmic decay: a relevant document at position 10 contributes less than one at position 1.

**MAP (Mean Average Precision):**
Average precision computed at each position where a relevant document appears, then averaged across all queries. Rewards systems that rank *all* relevant documents highly, not just the first one.

**Context Hit Rate (RAG-specific):**
```
Hit Rate = |queries where retrieved context contains the answer| / |total queries|
```
The most practical metric for RAG: does your retrieval pipeline give the LLM what it needs to answer correctly? This is what ultimately matters — not abstract IR metrics, but whether the generation succeeds.

---

## 5.13 Summary & Interview Tips

**Key Formulas to Know Cold:**
1. BM25: the full formula with k1=1.2, b=0.75
2. Cosine similarity: dot product of unit-normalized vectors
3. RRF: weight / (k + rank), with k=60

**Common Interview Questions and Strong Answers:**

*"BM25 vs dense retrieval?"*
The answer is *both*. BM25 catches exact matches (IDs, names, code symbols) that embedding models miss. Dense retrieval catches semantic matches (paraphrases, synonyms, intent) that BM25 misses. Use hybrid retrieval with RRF to combine them. Weight dense slightly higher (0.6) because most queries are natural language, but never drop BM25.

*"How do you evaluate retrieval?"*
Recall@k for coverage (are we finding all relevant documents?), MRR for ranking quality (is the best result near the top?), and context hit rate for end-to-end RAG evaluation (does the retrieved context enable correct generation?).

*"How do you choose chunk size?"*
200-500 tokens with semantic boundaries. Too small and you lose context; too large and you dilute the embedding signal. Use sliding window overlap (50 tokens) to prevent boundary losses. For tables, chunk per row with headers injected.

*"Why not just use a cross-encoder for everything?"*
Computational cost. A cross-encoder requires a forward pass per query-document pair. For 100K documents, that is 100K forward passes per query. Use bi-encoder for recall (top-100), cross-encoder for precision (re-rank to top-5).

**Practical Starting Recipe:**
1. Embedding model: all-MiniLM-L6-v2 (fast, good quality, small memory)
2. Sparse retrieval: BM25 via your vector DB's built-in FTS
3. Fusion: RRF with ann_weight=0.6, bm25_weight=0.4
4. Re-ranking: ms-marco-MiniLM-L-6-v2 cross-encoder on top-20
5. Chunking: 300 tokens, semantic boundaries, 50-token overlap
6. Vector DB: LanceDB for production, ChromaDB for prototypes

Start here. Measure with recall@10 and context hit rate. Upgrade components only when metrics show specific weaknesses. Premature optimization in retrieval is just as wasteful as in code.

---

*Next chapter: we build on these retrieval foundations to construct full RAG pipelines — orchestrating retrieval, re-ranking, context assembly, and generation into a coherent system.*
