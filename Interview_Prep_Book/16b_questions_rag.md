# RAG & Information Retrieval Interview Questions (Q81-Q110)

---

### Q81: What is TF-IDF?

**Answer:** TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that evaluates how relevant a word is to a document within a corpus. The formula is $\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$, where $\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$ is the normalized term frequency and $\text{IDF}(t, D) = \log\frac{|D|}{|\{d \in D : t \in d\}|}$ penalizes terms that appear in many documents. Words that appear frequently in a specific document but rarely across the corpus receive high TF-IDF scores, making them strong discriminators. It forms the foundation of classical information retrieval before neural methods.

**In Practice:** In the SSV RAG system, TF-IDF concepts underpin the BM25 component of the hybrid retrieval pipeline, providing lexical matching that catches exact terminology the dense model might miss.

---

### Q82: Explain the BM25 formula (k1=1.2, b=0.75).

**Answer:** BM25 (Best Matching 25) is a probabilistic ranking function that scores document relevance given a query. The formula is $\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot (1 - b + b \cdot \frac{|d|}{avgdl})}$, where $f_{t,d}$ is term frequency in document $d$, $|d|$ is document length, and $avgdl$ is average document length in the corpus. The parameter $k_1 = 1.2$ controls term frequency saturation (how quickly additional occurrences stop mattering), while $b = 0.75$ controls document length normalization (higher values penalize longer documents more). The IDF component is typically computed as $\log\frac{N - n(t) + 0.5}{n(t) + 0.5}$ where $N$ is total documents and $n(t)$ is documents containing term $t$.

**In Practice:** The SSV RAG hybrid pipeline uses BM25 with weight 0.4 in the RRF fusion, providing robust lexical matching for exact regulatory terminology that dense embeddings might map to nearby but incorrect concepts.

---

### Q83: Why does BM25 have term frequency saturation?

**Answer:** Term frequency saturation means that after a certain number of occurrences, additional repetitions of a term contribute diminishing returns to the relevance score. This is controlled by the $k_1$ parameter: the scoring function asymptotically approaches $(k_1 + 1)$ as term frequency increases, preventing keyword-stuffed documents from dominating results. Without saturation (as in raw TF-IDF), a document mentioning "compliance" 100 times would score disproportionately higher than one mentioning it 10 times, even though both are clearly about compliance. This models the intuition that a term's presence matters more than its exact count beyond a reasonable threshold. The bounded behavior also improves robustness against adversarial or redundant content.

**In Practice:** In the SSV RAG system, BM25 saturation ensures that regulatory documents repeating standard terminology (like "validation" or "qualification") are not artificially boosted over genuinely relevant passages.

---

### Q84: What is dense retrieval and how does it differ from BM25?

**Answer:** Dense retrieval encodes both queries and documents into continuous vector representations (embeddings) in a shared semantic space, then retrieves documents by vector similarity (typically cosine or dot product). Unlike BM25 which operates on exact lexical matching and bag-of-words statistics, dense retrieval captures semantic relationships: "car" and "automobile" would have high similarity even with zero token overlap. Dense models are trained on large corpora of query-passage pairs (like MS MARCO) to learn these semantic mappings. However, dense retrieval can struggle with rare terms, exact keyword matching, and out-of-domain generalization where BM25 remains robust. The computational cost shifts from index-time (inverted index) to encoding time (neural forward pass) plus approximate nearest neighbor search.

**In Practice:** The SSV RAG system combines dense retrieval (ANN with weight 0.6) with BM25 (weight 0.4) via RRF to capture both semantic understanding and precise lexical matches across pharmaceutical compliance documents.

---

### Q85: Explain bi-encoder architecture for retrieval.

**Answer:** A bi-encoder uses two independent transformer encoders (often sharing weights) to separately encode the query and document into fixed-size dense vectors, which are then compared using a similarity function like cosine similarity or dot product. The key advantage is that document embeddings can be pre-computed and indexed offline, making retrieval extremely fast at query time since only the query needs encoding. Training typically uses contrastive loss: positive pairs (query, relevant passage) should have high similarity while negative pairs should have low similarity, with in-batch negatives and hard negatives improving training efficiency. The architecture trades accuracy for speed because the query and document cannot attend to each other during encoding. Models like all-MiniLM-L6-v2 and sentence-transformers produce 384-768 dimensional vectors suitable for ANN indexing.

**In Practice:** The SPOT CHECK system uses all-MiniLM-L6-v2 as a bi-encoder to embed sub-points and retrieve the top_k=5 most relevant context chunks from ChromaDB per session.

---

### Q86: What is cosine similarity? (formula and geometric interpretation)

**Answer:** Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space, defined as $\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}||\mathbf{B}|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}$. Geometrically, it measures directional alignment regardless of magnitude: vectors pointing in the same direction have similarity 1, orthogonal vectors have 0, and opposite vectors have -1. This magnitude invariance is crucial for embeddings because it means document length does not bias similarity scores. In high-dimensional embedding spaces (384+ dimensions), most random vectors are nearly orthogonal, so meaningful similarities typically fall in the 0.3-1.0 range. Cosine similarity is equivalent to dot product on L2-normalized vectors, which is why many systems normalize embeddings at index time.

**In Practice:** The SSV RAG semantic cache uses cosine similarity with a threshold of 0.92 to determine whether an incoming query is semantically equivalent to a previously cached query, enabling the 5.5x latency reduction.

---

### Q87: Describe the all-MiniLM-L6-v2 model (384-dim, training).

**Answer:** all-MiniLM-L6-v2 is a sentence-transformer model that produces 384-dimensional embeddings, distilled from a larger model into a 6-layer MiniLM architecture for efficiency. It was trained in two stages: first, knowledge distillation from a larger teacher model on 1 billion sentence pairs, then fine-tuning on over 1 billion training pairs from diverse sources using contrastive learning with in-batch negatives. The "all" prefix indicates training on a concatenation of all available sentence-similarity datasets, giving it broad domain coverage. It achieves a strong balance between embedding quality and inference speed, with encoding throughput roughly 4x faster than 12-layer models while retaining approximately 95% of the performance. The 384-dimensional output keeps storage requirements low while maintaining sufficient expressiveness for semantic similarity tasks.

**In Practice:** SPOT CHECK uses all-MiniLM-L6-v2 to embed both sub-point descriptions and document chunks into ChromaDB, enabling fast per-session semantic retrieval with top_k=5 at minimal compute cost.

---

### Q88: What is a cross-encoder and why is it more accurate than bi-encoder?

**Answer:** A cross-encoder takes both the query and document as a single concatenated input to a transformer, allowing full bidirectional attention between all tokens of both texts simultaneously. This produces a single relevance score rather than separate embeddings, enabling the model to capture fine-grained interactions like negation, entity relationships, and contextual disambiguation that bi-encoders miss. The accuracy advantage comes from this joint encoding: in a bi-encoder, the query representation cannot attend to document tokens during encoding, forcing each side to compress all meaning into a fixed vector independently. However, cross-encoders cannot pre-compute document representations, making them $O(N)$ per query where $N$ is the corpus size, so they are impractical for first-stage retrieval over large collections. They are therefore used as rerankers on a small candidate set (typically 20-100 documents).

**In Practice:** The SSV RAG pipeline uses the ms-marco-MiniLM-L-6-v2 cross-encoder to rerank candidates retrieved by the hybrid BM25+ANN first stage, significantly improving precision in the final top results.

---

### Q89: Explain ms-marco-MiniLM-L-6-v2 (what it is trained on).

**Answer:** ms-marco-MiniLM-L-6-v2 is a cross-encoder model fine-tuned on the MS MARCO (Microsoft MAchine Reading COmprehension) passage ranking dataset, which contains approximately 500K real Bing search queries paired with human-annotated relevant and non-relevant passages from web documents. The model is built on MiniLM-L-6 (6 transformer layers, 22M parameters) for fast inference while maintaining high ranking quality. Training uses binary cross-entropy loss on (query, passage, label) triples where labels indicate relevance, with hard negatives mined from BM25 results to teach the model to distinguish subtle relevance differences. It achieves state-of-the-art performance on passage reranking benchmarks relative to its size class. The model outputs a single logit score representing passage relevance to the query, with higher scores indicating greater relevance.

**In Practice:** In the SSV RAG system, ms-marco-MiniLM-L-6-v2 serves as the reranking cross-encoder, rescoring the top candidates from hybrid retrieval to ensure the most relevant pharmaceutical compliance passages are presented to the LLM for answer generation.

---

### Q90: What is HNSW? (how it builds the graph, search algorithm)

**Answer:** HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor algorithm that builds a multi-layer graph where each layer is a navigable small world network with decreasing density from bottom to top. During construction, each new element is inserted by finding its nearest neighbors via greedy search starting from the top layer, then establishing bidirectional connections at each layer it is assigned to (layer assignment follows an exponential decay probability). The search algorithm starts at the entry point in the top (sparsest) layer, greedily traverses to the nearest node, then descends to the next layer using that node as the starting point, repeating until reaching the bottom layer where it performs a more thorough beam search with a configurable $ef$ parameter. The hierarchical structure provides $O(\log N)$ search complexity with high recall, and the small-world property ensures short paths between any two nodes.

**In Practice:** The SSV RAG system uses HNSW indexing in LanceDB for the dense ANN retrieval component, providing sub-millisecond approximate nearest neighbor search across the pharmaceutical compliance document embeddings.

---

### Q91: Compare HNSW vs IVF for ANN search.

**Answer:** HNSW builds a persistent graph structure with $O(N \log N)$ construction time and $O(N \cdot M)$ memory (where $M$ is the max connections per node), offering consistently high recall (>95%) with $O(\log N)$ query time and no training phase. IVF (Inverted File Index) partitions the vector space into clusters using k-means, requires a training step on representative data, uses less memory since it only stores centroids plus inverted lists, and has query time proportional to the number of probed clusters ($nprobe$). HNSW excels when memory is available and high recall is critical, while IVF is preferred for very large-scale deployments (billions of vectors) where memory constraints dominate. HNSW supports dynamic insertion without rebuilding, whereas IVF performance degrades as the data distribution drifts from the training set. IVF can be combined with product quantization (IVF-PQ) for extreme compression.

**In Practice:** The SSV RAG system chose HNSW via LanceDB for its pharmaceutical document corpus because the dataset size (thousands, not billions) prioritizes recall quality and dynamic updates over memory optimization.

---

### Q92: Why use hybrid retrieval (BM25 + dense)?

**Answer:** Hybrid retrieval combines lexical (BM25) and semantic (dense) search to compensate for each method's weaknesses. BM25 excels at exact keyword matching, rare terms, and out-of-domain robustness but misses synonyms and paraphrases; dense retrieval captures semantic similarity but can fail on exact terminology, acronyms, and domain-specific jargon it was not trained on. Empirically, hybrid approaches consistently outperform either method alone across benchmarks like BEIR, with gains of 5-15% in recall@k depending on the domain. The combination is especially valuable in specialized domains where vocabulary is precise (medical, legal, regulatory) and both exact matching and conceptual understanding matter. Fusion methods like RRF or linear score combination merge the ranked lists into a unified result.

**In Practice:** The SSV RAG system implements hybrid retrieval with ANN weight 0.6 and BM25 weight 0.4 fused via RRF (k=60), achieving the 77% answer rate on pharmaceutical compliance queries that neither method alone could match.

---

### Q93: Explain Reciprocal Rank Fusion (formula: 1/(k+rank), k=60).

**Answer:** Reciprocal Rank Fusion (RRF) combines multiple ranked lists by assigning each document a score based on its rank position in each list: $\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$, where $R$ is the set of ranked lists and $k$ is a constant (typically 60) that mitigates the impact of high rankings from a single source. Documents are then sorted by their combined RRF score. The parameter $k=60$ was empirically determined by Cormack et al. (2009) to provide robust fusion across diverse retrieval systems without requiring score normalization. RRF is elegant because it requires no tuning of score scales, no training data for weight learning, and remains robust when one ranker produces poor results (a low rank simply contributes a small score). Documents appearing in multiple lists accumulate score from each, naturally boosting results that multiple methods agree are relevant.

**In Practice:** The SSV RAG hybrid pipeline applies RRF with k=60 to fuse BM25 (weight 0.4) and ANN (weight 0.6) ranked lists, providing a parameter-free merging strategy that handles the different score distributions of lexical and dense retrieval.

---

### Q94: Why does RRF use rank instead of score?

**Answer:** RRF uses rank instead of raw scores because scores from different retrieval systems are fundamentally incomparable: BM25 scores are unbounded positive values dependent on corpus statistics, while cosine similarities are bounded in [-1, 1], and different dense models produce different score distributions. Normalizing scores (min-max, z-score) requires knowing the full score distribution per query, which is expensive and unstable with small candidate sets. Ranks, however, are uniformly distributed integers that provide a common currency across any retrieval method regardless of its internal scoring mechanism. This rank-based approach also provides robustness to outliers: a single extremely high-scoring document in one system cannot dominate the fused results. The simplicity of rank-based fusion eliminates the need for learned fusion weights or calibration data.

**In Practice:** In the SSV RAG system, using rank-based RRF means the pipeline does not need to calibrate BM25 scores against cosine similarity scores from the ANN index, simplifying the hybrid retrieval architecture.

---

### Q95: Compare ChromaDB vs LanceDB vs FAISS vs Pinecone.

**Answer:** ChromaDB is a lightweight, Python-native embedded vector database optimized for prototyping and small-to-medium workloads with automatic embedding generation and simple metadata filtering. LanceDB is a serverless, embedded columnar vector database built on Lance format that supports disk-based indexing, multimodal data, and efficient updates without a separate server process. FAISS (Facebook AI Similarity Search) is a low-level C++ library focused purely on efficient similarity search algorithms (IVF, HNSW, PQ) without database features like persistence, metadata, or CRUD operations. Pinecone is a fully managed cloud vector database offering horizontal scaling, high availability, and serverless tiers but incurs ongoing costs and network latency. The choice depends on scale, operational complexity tolerance, and whether you need managed infrastructure or prefer embedded/local solutions.

**In Practice:** The SSV RAG system uses LanceDB for its serverless, disk-efficient architecture suitable for production pharmaceutical retrieval, while SPOT CHECK uses ChromaDB for its simplicity in per-session ephemeral vector stores during compliance checks.

---

### Q96: When would you choose LanceDB over ChromaDB?

**Answer:** LanceDB is preferred when you need disk-based storage for larger-than-memory datasets, efficient columnar access patterns for metadata-heavy workloads, or zero-copy integration with ML frameworks via the Lance format. Its append-only, versioned storage model provides better crash recovery and supports time-travel queries. LanceDB handles multimodal data (images, text, structured) natively in a unified table format, whereas ChromaDB is primarily text-embedding focused. For production workloads requiring updates to millions of vectors without full reindexing, LanceDB's architecture is more suitable. ChromaDB is better for rapid prototyping, small datasets that fit in memory, and scenarios where its built-in embedding function support simplifies the pipeline.

**In Practice:** The SSV RAG system chose LanceDB because the pharmaceutical compliance corpus requires persistent, versioned storage with efficient disk-based retrieval and the ability to update regulatory documents without rebuilding the entire index.

---

### Q97: What are chunking strategies? (fixed, semantic, table-aware)

**Answer:** Fixed-size chunking splits documents into equal-length segments (e.g., 512 tokens) with optional overlap (e.g., 50-100 tokens) to maintain context at boundaries; it is simple but can split sentences or concepts mid-thought. Semantic chunking uses sentence embeddings to detect topic boundaries: consecutive sentences with cosine similarity below a threshold are split into separate chunks, preserving coherent ideas at variable lengths. Table-aware chunking recognizes structured elements (tables, lists, code blocks) and keeps them intact as atomic units, preventing retrieval of partial tables that lose meaning. Recursive chunking (used in LangChain) attempts splits at paragraph boundaries first, then sentences, then characters, maintaining natural units where possible. The choice of strategy significantly impacts retrieval quality: poor chunking creates fragments that are either too vague for matching or too incomplete for answering.

**In Practice:** The SSV RAG system uses table-aware and semantic chunking strategies to handle pharmaceutical compliance documents that contain structured regulatory tables, ensuring complete regulatory requirements are preserved as retrievable units.

---

### Q98: What is the ideal chunk size and why?

**Answer:** The ideal chunk size for most RAG systems is 256-512 tokens, balancing specificity (smaller chunks match queries more precisely) against context sufficiency (larger chunks provide enough information to answer questions). Empirical studies show that chunks below 128 tokens often lack sufficient context for meaningful answers, while chunks above 1024 tokens dilute relevance signals and waste LLM context window space. The optimal size depends on the embedding model's training: all-MiniLM-L6-v2 was trained on sentences/short paragraphs (optimal around 256 tokens), while models like text-embedding-3-large handle longer contexts. Overlap of 10-20% between chunks (50-100 tokens) prevents information loss at boundaries. The chunk size should also align with the expected answer granularity: factoid QA benefits from smaller chunks while summarization tasks benefit from larger ones.

**In Practice:** The SSV RAG system tunes chunk size to match pharmaceutical regulatory content structure, where individual requirements and guidance statements typically span 200-400 tokens and need to be retrieved as complete, actionable units.

---

### Q99: Explain semantic caching (cosine threshold approach).

**Answer:** Semantic caching stores previous query-response pairs and uses embedding similarity to determine if a new query is semantically equivalent to a cached one, avoiding redundant retrieval and LLM calls. The process works as follows: embed the incoming query, compute cosine similarity against all cached query embeddings, and if the maximum similarity exceeds a threshold (e.g., 0.92), return the cached response directly. Unlike exact-match caching (which only hits on identical strings), semantic caching handles paraphrases: "What is the validation requirement?" and "Tell me about validation requirements" would be cache hits. The cache typically uses a vector index itself for efficient lookup, with TTL (time-to-live) policies to handle stale information. This approach dramatically reduces latency and cost for repetitive or near-repetitive queries common in production systems.

**In Practice:** The SSV RAG system implements semantic caching with a cosine threshold of 0.92, achieving a 5.5x latency reduction by serving cached responses for paraphrased queries about frequently asked pharmaceutical compliance topics.

---

### Q100: Why is the semantic cache threshold 0.92 and not higher/lower?

**Answer:** The threshold of 0.92 represents an empirically tuned balance between cache hit rate and answer accuracy. Below 0.90, semantically different queries start matching (e.g., "GMP requirements for APIs" vs "GMP requirements for finished products" might have similarity ~0.89), causing incorrect cached responses to be served. Above 0.95, only near-identical phrasings match, reducing the cache hit rate to near that of exact-match caching and eliminating the benefit of semantic matching. The 0.92 sweet spot was determined through evaluation on representative query pairs, measuring the precision of cache hits (fraction that a human would consider equivalent queries). Domain specificity matters: in pharmaceutical compliance where terminology precision is critical, a higher threshold prevents dangerous conflation of related but distinct regulatory concepts.

**In Practice:** The SSV RAG system validated the 0.92 threshold through its 100-query evaluation benchmark, confirming it maximizes cache utilization while maintaining answer correctness across pharmaceutical compliance queries.

---

### Q101: What is recall@k and how do you measure it?

**Answer:** Recall@k measures the proportion of all relevant documents that appear in the top-k retrieved results: $\text{Recall@k} = \frac{|\text{relevant documents in top-k}|}{|\text{total relevant documents}|}$. It answers the question "of everything I should have found, how much did I actually find in my top-k?" To measure it, you need a ground truth set of relevant documents for each query (typically from human annotations or click data). For example, if 5 documents are relevant to a query and 3 appear in the top-10 results, Recall@10 = 0.6. In RAG systems, high recall@k is critical because the LLM can only generate correct answers from retrieved context; missed relevant passages directly cause answer failures. It is typically measured at k values matching the retrieval pipeline's cutoff (e.g., k=10, 20, 50).

**In Practice:** The SSV RAG system's 100-query evaluation benchmark measures recall across the hybrid retrieval pipeline to ensure the BM25+ANN fusion captures sufficient relevant passages before cross-encoder reranking.

---

### Q102: Explain MRR (Mean Reciprocal Rank).

**Answer:** Mean Reciprocal Rank (MRR) measures how quickly a system returns the first relevant result, computed as $\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$, where $\text{rank}_i$ is the position of the first relevant document for query $i$. If the first relevant result is at position 1, the reciprocal rank is 1.0; at position 2 it is 0.5; at position 3 it is 0.33, and so on. MRR is ideal for evaluating systems where users primarily care about the single best answer (like question answering), rather than browsing multiple results. It ranges from 0 to 1, where 1.0 means every query's first relevant document appears at rank 1. MRR penalizes systems that bury relevant results deeper in the list but does not account for the total number of relevant documents retrieved.

**In Practice:** The SSV RAG system uses MRR alongside recall to evaluate whether the hybrid retrieval plus cross-encoder reranking pipeline consistently places the most relevant pharmaceutical compliance passage at the top of results.

---

### Q103: What is NDCG? (formula with discounted gain)

**Answer:** NDCG (Normalized Discounted Cumulative Gain) evaluates ranked result quality by considering both relevance grades and position, defined as $\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$, where $\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$ and IDCG@k is the DCG of the ideal (perfectly sorted) ranking. The logarithmic discount $\frac{1}{\log_2(i+1)}$ reduces the contribution of results at lower positions, modeling the intuition that users are less likely to examine results further down the list. Unlike binary metrics, NDCG supports graded relevance (e.g., 0=irrelevant, 1=partially relevant, 2=highly relevant), making it suitable for nuanced evaluation. Normalization by IDCG ensures scores are comparable across queries with different numbers of relevant documents, producing values in [0, 1] where 1.0 represents a perfect ranking.

**In Practice:** The SSV RAG evaluation benchmark uses NDCG to assess whether the cross-encoder reranking stage correctly prioritizes highly relevant pharmaceutical passages over partially relevant ones.

---

### Q104: How do you evaluate a RAG system end-to-end?

**Answer:** End-to-end RAG evaluation requires measuring both retrieval quality and generation quality independently and jointly. Retrieval metrics (Recall@k, MRR, NDCG) assess whether the right context was found, while generation metrics assess answer quality given that context: faithfulness (does the answer only use provided context?), relevance (does it address the query?), and completeness (does it cover all aspects?). Automated evaluation uses LLM-as-judge (GPT-4 scoring faithfulness/relevance on 1-5 scales), RAGAS framework metrics, or reference-based metrics like ROUGE/BERTScore against gold answers. Human evaluation remains the gold standard for production systems, using annotator agreement on correctness and hallucination detection. A/B testing with user satisfaction signals (thumbs up/down, follow-up query rate) provides real-world validation.

**In Practice:** The SSV RAG system uses a 100-query evaluation benchmark measuring retrieval recall, reranking precision, answer correctness (77% answer rate), and response latency to validate the full pipeline from query to generated answer.

---

### Q105: What is query expansion?

**Answer:** Query expansion enriches the original user query with additional terms or reformulations to improve retrieval recall, addressing the vocabulary mismatch problem where users and documents use different words for the same concepts. Techniques include pseudo-relevance feedback (PRF) where top-k initial results are used to extract expansion terms, LLM-based expansion (asking an LLM to generate hypothetical document passages or synonyms), and acronym/abbreviation expansion from domain dictionaries. HyDE (Hypothetical Document Embeddings) generates a hypothetical answer using an LLM, then uses that answer's embedding for retrieval since it is closer to document language than the question. Multi-query approaches generate multiple query variants and union the results. Query expansion is particularly valuable in domains with specialized vocabulary where users may not know exact document terminology.

**In Practice:** The SSV RAG system benefits from query expansion in handling pharmaceutical compliance queries where users might ask about "CSV" (Computerized System Validation) using varied terminology that BM25 alone would miss.

---

### Q106: Explain context compression for RAG (dedup, truncation).

**Answer:** Context compression reduces the retrieved passages before feeding them to the LLM, minimizing token usage, cost, and noise while preserving relevant information. Deduplication removes near-identical or overlapping passages that arise from chunk overlap or multiple retrieval paths finding the same content, using either exact string matching or semantic similarity thresholds. Truncation removes irrelevant sentences within passages using extractive summarization or an LLM-based compressor that keeps only query-relevant sentences. LongLLMLingua and similar methods score each token's relevance to the query using perplexity, removing low-information tokens. Additional techniques include reordering (placing most relevant passages first due to "lost in the middle" phenomena) and merging adjacent chunks back into coherent passages.

**In Practice:** The SSV RAG pipeline applies deduplication after RRF fusion since BM25 and ANN often retrieve overlapping passages, reducing redundant context before the cross-encoder reranking stage.

---

### Q107: How do you handle out-of-domain queries (guardrails)?

**Answer:** Out-of-domain (OOD) detection prevents RAG systems from generating hallucinated answers when queries fall outside the knowledge base's coverage. Approaches include: retrieval confidence thresholding (if the maximum retrieval score is below a calibrated threshold, abstain from answering), embedding-based OOD detection (measuring distance from query embedding to the corpus centroid or nearest cluster), and classifier-based routing that categorizes queries as in-domain/out-of-domain before retrieval. The system should respond with "I don't have information about this topic" rather than fabricating answers from tangentially related passages. Additional guardrails include topic classification (rejecting queries outside defined categories), input validation (blocking prompt injection attempts), and output validation (checking generated answers for unsupported claims against retrieved context).

**In Practice:** The SSV RAG system's 77% answer rate implicitly reflects its guardrail approach: 23% of queries receive abstention responses when retrieval confidence is insufficient, preventing hallucinated pharmaceutical compliance guidance.

---

### Q108: What is the reranking pipeline? (retrieve many, rerank few)

**Answer:** The reranking pipeline follows a "telescoping" architecture: first retrieve a large candidate set (50-200 documents) using fast but less accurate methods (BM25, bi-encoder ANN), then rerank a smaller subset (10-50) using a slower but more accurate cross-encoder model. This two-stage approach is necessary because cross-encoders have $O(N \cdot L)$ complexity (processing each query-document pair through the full transformer), making them infeasible for the entire corpus. The first stage optimizes for recall (finding all potentially relevant documents), while the reranker optimizes for precision (correctly ordering by relevance). Some systems add a third stage with even more expensive models (e.g., larger cross-encoders or LLM-based rerankers) on the top 5-10 candidates. The retrieve-then-rerank pattern reduces latency from seconds to milliseconds while maintaining high result quality.

**In Practice:** The SSV RAG system retrieves candidates via hybrid BM25+ANN, fuses with RRF, then reranks the top candidates with ms-marco-MiniLM-L-6-v2 cross-encoder, achieving high precision within acceptable latency constraints.

---

### Q109: How does connection pooling improve RAG latency?

**Answer:** Connection pooling maintains a pool of pre-established connections to databases, vector stores, and LLM APIs, eliminating the overhead of creating new connections per request (TCP handshake, TLS negotiation, authentication). For RAG systems, this impacts three critical paths: vector database connections (LanceDB/ChromaDB client initialization), LLM API connections (HTTP/2 persistent connections to inference endpoints), and any metadata store connections (PostgreSQL, Redis). Without pooling, each query incurs 50-200ms of connection setup latency across these services. Pool configuration parameters (min/max connections, idle timeout, health checks) must balance resource usage against latency: too few connections create queuing under load, while too many waste memory and may hit server-side limits. Connection pooling also enables connection reuse across concurrent requests, critical for serving multiple users simultaneously.

**In Practice:** The SSV RAG system uses connection pooling to maintain persistent connections to LanceDB and the LLM inference endpoint, contributing to the overall latency optimization that achieves 5.5x speedup alongside semantic caching.

---

### Q110: Design a caching strategy for a RAG system (3 tiers).

**Answer:** A three-tier caching strategy for RAG optimizes for different access patterns: Tier 1 (Semantic Cache) stores complete query-response pairs keyed by query embeddings with cosine similarity matching (threshold ~0.92), providing instant responses for repeated or paraphrased questions with TTL based on source document update frequency. Tier 2 (Retrieval Cache) caches the retrieval results (chunk IDs and scores) for recent queries, bypassing the vector search and BM25 computation for queries that are similar but might need fresh LLM generation (e.g., when prompt templates change). Tier 3 (Embedding Cache) caches computed embeddings for documents and frequent query patterns, avoiding redundant encoder forward passes for unchanged content. Invalidation strategies differ per tier: Tier 1 invalidates when source documents update, Tier 2 invalidates on index changes, and Tier 3 invalidates only when the embedding model changes. This layered approach provides graceful degradation where each tier independently reduces latency.

**In Practice:** The SSV RAG system implements the semantic cache tier (Tier 1) at a 0.92 cosine threshold, achieving 5.5x latency reduction; the architecture supports extending to retrieval and embedding cache tiers as query volume scales.
