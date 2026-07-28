# Interview Questions: Large Language Models (Q51-Q80)

---

### Q51: What is tokenization? (BPE algorithm)

**Answer:** Tokenization is the process of converting raw text into a sequence of discrete integer IDs that a language model can process. Byte-Pair Encoding (BPE) is the dominant tokenization algorithm: it starts with individual bytes or characters as an initial vocabulary, then iteratively merges the most frequent adjacent pair into a new token until a target vocabulary size $|V|$ is reached. The final vocabulary balances expressiveness (common words become single tokens) with coverage (rare words decompose into subword units). Typical vocabulary sizes range from 32K (GPT-2) to 200K (Claude 3), directly affecting embedding matrix dimensions $\mathbf{E} \in \mathbb{R}^{|V| \times d}$ and thus model size and memory.

**In Practice:** In MTP, comparing 0.5B vs 8B vs 14B models required understanding that all three share the same tokenizer vocabulary, so token-level branching logic remained consistent across model sizes.

---

### Q52: How does Byte-Pair Encoding work step by step?

**Answer:** BPE begins by splitting text into individual characters (or bytes) and counting all adjacent pairs across the corpus. The most frequent pair is merged into a new symbol, the corpus representation is updated, and pair frequencies are recalculated. This merge step repeats for $N$ iterations (where $N$ = desired vocab size minus initial alphabet size), building a deterministic merge table. At inference time, the same ordered merge rules are applied greedily to unseen text, guaranteeing that any input can be encoded without out-of-vocabulary errors. The compression ratio (characters per token) typically lands between 3.5 and 4.5 for English text.

**In Practice:** In iFAST, understanding BPE tokenization was essential for estimating prompt lengths when designing the 14-tool system prompt to stay within Claude's context budget.

---

### Q53: What is SentencePiece and how does it differ from BPE?

**Answer:** SentencePiece is a language-agnostic tokenization library that treats the input as a raw byte stream (including whitespace) rather than pre-tokenized words, eliminating the need for language-specific pre-processing. While it supports BPE internally, it also offers a Unigram model that assigns probabilities to all candidate subwords and iteratively prunes the vocabulary by removing tokens whose removal least increases the corpus likelihood under the unigram language model $P(x) = \prod_{i} P(x_i)$. This makes SentencePiece particularly effective for multilingual models because it handles scripts without explicit word boundaries (Chinese, Japanese, Thai). The Unigram approach also enables multiple valid segmentations, useful for regularization during training.

**In Practice:** In SSV RAG, both Claude and Gemini backends use SentencePiece-family tokenizers, so token count estimates needed model-specific tiktoken/tokenizer libraries for accurate budget calculations.

---

### Q54: Explain autoregressive generation (next-token prediction)

**Answer:** Autoregressive generation produces text one token at a time, where each new token is conditioned on all previously generated tokens: $P(x_t | x_1, x_2, \ldots, x_{t-1})$. The model computes a probability distribution over the entire vocabulary at each step via a softmax over logits, selects a token according to a sampling strategy, appends it to the sequence, and repeats until a stop token or maximum length is reached. This sequential dependency means generation latency scales linearly with output length $O(n)$, though each forward pass through the transformer is parallelizable across layers. The causal mask in the attention mechanism ensures the model never attends to future positions, maintaining the autoregressive property.

**In Practice:** In MTP, autoregressive generation with temperature-controlled branching at $T \in \{0.15, 0.45, 0.75\}$ produced diverse reasoning paths that were then scored and pruned to find optimal solutions.

---

### Q55: What is temperature in LLM sampling? (formula: softmax(logits/T))

**Answer:** Temperature $T$ is a scalar that reshapes the probability distribution before sampling by dividing logits by $T$ before applying softmax: $P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$. When $T \to 0$, the distribution collapses to a point mass on the highest-logit token (greedy decoding); when $T = 1$, the original model distribution is preserved; when $T > 1$, the distribution flattens, increasing randomness and diversity. Temperature does not change which token has the highest probability, only the relative mass allocated to lower-probability alternatives. Choosing the right temperature requires balancing factual accuracy (low $T$) against creative diversity (high $T$).

**In Practice:** In MTP, three temperature tiers $[0.15, 0.45, 0.75]$ were used deliberately: $T=0.15$ for near-deterministic factual steps, $T=0.45$ for moderate exploration, and $T=0.75$ for creative divergent branching.

---

### Q56: Explain top-k sampling

**Answer:** Top-k sampling restricts the candidate set at each generation step to only the $k$ tokens with the highest logits, zeroing out the probabilities of all other tokens and renormalizing: $P'(x_i) = P(x_i) / \sum_{j \in \text{top-}k} P(x_j)$ for $i \in \text{top-}k$, else $0$. This prevents the model from sampling extremely unlikely tokens that could derail coherence, while still allowing diversity within the top candidates. The fixed $k$ is a weakness: for peaked distributions (e.g., after "The capital of France is"), $k=50$ still includes many irrelevant tokens; for flat distributions, $k=50$ might cut off plausible continuations. Typical values range from $k=10$ for focused tasks to $k=100$ for creative generation.

**In Practice:** In MTP, top-k was combined with temperature to control the branching factor, ensuring that even at $T=0.75$ the model sampled from semantically plausible continuations rather than noise.

---

### Q57: Explain nucleus (top-p) sampling

**Answer:** Nucleus sampling (top-p) dynamically selects the smallest set of tokens whose cumulative probability exceeds a threshold $p$: $\text{top-}p = \min \{S \subseteq V : \sum_{x_i \in S} P(x_i) \geq p\}$, then renormalizes within that set. Unlike top-k which uses a fixed count, top-p adapts to the shape of the distribution -- it includes fewer tokens when the model is confident and more when uncertain. This makes it more robust across diverse generation contexts without hyperparameter tuning per prompt. Common values are $p \in [0.9, 0.95]$ for general use, with lower values like $p=0.7$ for more deterministic outputs. It is often combined with temperature: first scale logits by $T$, then apply the top-p filter.

**In Practice:** In SSV RAG, nucleus sampling with $p=0.9$ was used alongside low temperature for the retrieval-augmented answers, ensuring grounded responses while allowing minor phrasing variation.

---

### Q58: What is the context window and why does it matter?

**Answer:** The context window is the maximum number of tokens (input + output combined) that a model can process in a single forward pass, determined by the positional encoding scheme and memory capacity. Attention complexity scales as $O(n^2)$ with context length $n$ in standard transformers, making longer contexts quadratically more expensive in compute and memory. Modern models use techniques like RoPE (Rotary Position Embedding) with NTK-aware interpolation to extend context from training lengths (e.g., 8K) to inference lengths (128K-200K). The context window defines the upper bound on how much information -- system prompts, conversation history, retrieved documents, and tool outputs -- can inform a single generation.

**In Practice:** In iFAST, the 14-tool system prompt consumed a significant portion of Claude's context, requiring careful prompt engineering and conversation summarization to avoid context window overflow during long multi-turn sessions.

---

### Q59: Explain KV-cache (what it stores, why it speeds up inference)

**Answer:** The KV-cache stores the key and value projection matrices from all previous tokens across all attention layers: $\mathbf{K}_{\text{cache}} \in \mathbb{R}^{L \times n \times d_k}$ and $\mathbf{V}_{\text{cache}} \in \mathbb{R}^{L \times n \times d_v}$ where $L$ is layer count and $n$ is sequence length so far. During autoregressive generation, instead of recomputing attention over the entire sequence at each step, only the new token's query vector attends against the cached keys and values, reducing per-token computation from $O(n \cdot d)$ to $O(d)$ for the projection. The trade-off is memory: KV-cache grows linearly with sequence length and batch size, often dominating GPU memory at long contexts (e.g., 200K context at fp16 can require 30+ GB of KV-cache alone for a 70B model). This is why techniques like GQA (Grouped-Query Attention) reduce KV-cache size by sharing key-value heads.

**In Practice:** In MTP's vLLM serving setup, KV-cache memory management via PagedAttention was critical for efficiently batching multiple concurrent branching paths without running out of GPU memory.

---

### Q60: What is chain-of-thought prompting?

**Answer:** Chain-of-thought (CoT) prompting instructs the model to produce intermediate reasoning steps before arriving at a final answer, mimicking human step-by-step problem solving. This is achieved either by including exemplars with explicit reasoning traces (few-shot CoT) or by simply appending "Let's think step by step" (zero-shot CoT). CoT improves performance on arithmetic, logic, and multi-hop reasoning tasks by decomposing complex problems into simpler sub-problems that the model can solve within a single forward pass. The effectiveness scales with model size -- models below ~10B parameters show minimal CoT benefit, suggesting it requires sufficient internal representation capacity. Empirically, CoT can improve accuracy by 10-40% on benchmarks like GSM8K compared to direct answer prompting.

**In Practice:** In SPOT CHECK, chain-of-thought prompting was embedded in the system prompt to force the model to reason through compliance checks before outputting structured verdicts, reducing hallucinated conclusions.

---

### Q61: Compare zero-shot vs few-shot prompting

**Answer:** Zero-shot prompting provides only a task instruction with no examples, relying entirely on the model's pre-trained knowledge and instruction tuning to infer the expected format and behavior. Few-shot prompting prepends $k$ input-output exemplars (typically $k \in [1, 5]$) that demonstrate the desired task pattern, leveraging in-context learning where the model implicitly learns the mapping $f: X \to Y$ from the examples without gradient updates. Few-shot generally outperforms zero-shot on structured tasks (classification, extraction) but uses more context tokens and can introduce bias if exemplars are not representative. The performance gap narrows with larger models and better instruction tuning -- Claude 3.5 Sonnet often matches few-shot accuracy in zero-shot mode. Cost-wise, few-shot multiplies input tokens by the number of examples plus the overhead of formatting.

**In Practice:** In iFAST, zero-shot prompting with detailed system instructions was preferred over few-shot to conserve context space for the 14 tool definitions while still achieving high accuracy on structured tool selection.

---

### Q62: What makes a good system prompt?

**Answer:** A good system prompt establishes role/persona, defines explicit constraints and output format, provides domain context, and includes anti-hallucination guardrails -- all while remaining concise enough to leave room for user input and model output within the context window. The structure should follow: identity/role, then capabilities/limitations, then behavioral rules, then output format specification, then examples if space permits. Critical elements include explicit instructions about what the model should NOT do (negative constraints reduce hallucination more effectively than positive-only framing) and clear escalation paths for uncertain cases. System prompts should be version-controlled and A/B tested because small wording changes can cause significant behavioral shifts. Token efficiency matters: every system prompt token is charged on every request in a session.

**In Practice:** In SPOT CHECK, the system prompt explicitly stated "Blank > Wrong" as an anti-hallucination principle, instructing the model to leave fields empty rather than guess, which reduced false positive compliance findings by over 40%.

---

### Q63: How does tool-use/function calling work in LLMs?

**Answer:** Tool-use enables LLMs to invoke external functions by generating structured calls (function name + arguments as JSON) that an orchestration layer executes and returns results for. The model is trained (via fine-tuning or RLHF) to recognize when a user query requires external capabilities, select the appropriate tool from a provided schema, emit a well-formed JSON call, and then incorporate the tool's response into its final answer. The flow is: user message $\to$ model decides to call tool $\to$ orchestrator executes function $\to$ result injected as a new message $\to$ model synthesizes final response. Tool schemas are typically provided in the system prompt using JSON Schema or similar notation, defining parameters, types, and descriptions that the model uses to construct valid calls.

**In Practice:** In iFAST, 14 tools were defined for Claude via Bedrock's tool-use API, enabling the model to query databases, check compliance rules, retrieve documents, and trigger workflows -- all orchestrated through structured JSON tool calls with Pydantic validation.

---

### Q64: What is structured output (JSON mode)?

**Answer:** Structured output constrains the model to emit valid JSON (or another schema) rather than free-form text, typically enforced through either API-level constraints (response_format = "json_object"), grammar-guided decoding that masks invalid tokens at each step, or post-hoc Pydantic validation with retry loops. Grammar-guided approaches modify the sampling distribution at each token: $P'(x_i) = 0$ if token $x_i$ would produce invalid JSON given the partial output so far, ensuring 100% structural validity. This is essential for downstream programmatic consumption -- unreliable JSON means broken pipelines. The trade-off is slightly reduced reasoning quality since the model must satisfy format constraints simultaneously with content generation, and some providers add latency for constrained decoding.

**In Practice:** In SPOT CHECK, Pydantic-enforced structured output guaranteed that every compliance assessment returned valid JSON with required fields (verdict, confidence, evidence), enabling reliable downstream aggregation without parsing failures.

---

### Q65: How does vLLM achieve high throughput? (PagedAttention)

**Answer:** vLLM achieves high throughput through PagedAttention, which borrows virtual memory concepts from operating systems to manage KV-cache. Instead of pre-allocating contiguous memory for each sequence's maximum possible length, PagedAttention divides KV-cache into fixed-size blocks (pages) and allocates them on-demand via a block table, similar to how OS page tables map virtual to physical addresses. This eliminates internal fragmentation (wasted pre-allocated space) and external fragmentation (gaps between sequences), achieving near-zero memory waste and enabling 2-4x higher batch sizes compared to naive implementations. Additionally, vLLM implements continuous batching (iteration-level scheduling) so new requests can begin processing immediately when slots free up, rather than waiting for an entire batch to complete. The combination yields 2-24x throughput improvement over HuggingFace Transformers.

**In Practice:** In MTP, vLLM with PagedAttention served the local models (0.5B/8B/14B) and enabled efficient parallel generation of multiple branching paths per query, critical for the multi-temperature sampling strategy.

---

### Q66: What is Ollama and when would you use it?

**Answer:** Ollama is a lightweight inference server that packages quantized LLMs (typically in GGUF format) with a simple REST API for local deployment on consumer hardware. It handles model downloading, quantization selection (Q4_K_M, Q5_K_M, etc.), memory management, and provides an OpenAI-compatible API endpoint, abstracting away the complexity of llama.cpp configuration. Ollama is ideal for development/testing, privacy-sensitive workloads, edge deployment, and cost reduction when cloud API latency or pricing is prohibitive. Its limitations include lower throughput than vLLM for production batched inference and restricted model sizes by available RAM/VRAM (a Q4 quantized 70B model requires ~40GB RAM). The Modelfile system allows custom system prompts, temperature, and stop tokens to be baked into a deployable model artifact.

**In Practice:** In MTP, Ollama served as the rapid prototyping backend for comparing Qwen models (0.5B vs 8B vs 14B) locally before deploying the selected configuration to vLLM for production-scale branching experiments.

---

### Q67: Explain GPTQ quantization

**Answer:** GPTQ (Generative Pre-trained Transformer Quantization) is a post-training quantization method that compresses model weights to lower bit-widths (typically 4-bit or 3-bit) while minimizing the layer-wise reconstruction error $\|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}\|_2^2$ using a calibration dataset. It processes weights column-by-column using an approximate Hessian $\mathbf{H} = 2\mathbf{X}\mathbf{X}^T$ to determine which weights can tolerate more quantization error, applying Optimal Brain Quantization (OBQ) principles at scale. GPTQ achieves this in a single pass through the model (minutes to hours), making it practical for large models where quantization-aware training would be prohibitively expensive. The resulting models maintain 95-99% of full-precision accuracy while reducing memory by 4x and enabling inference on consumer GPUs. Group-wise quantization (e.g., group_size=128) further improves accuracy by maintaining separate scaling factors for blocks of weights.

**In Practice:** In MTP, GPTQ-quantized versions of the 14B model were benchmarked to determine whether 4-bit quantization preserved sufficient reasoning quality for the multi-path generation task, informing deployment memory requirements.

---

### Q68: What is AWQ quantization?

**Answer:** Activation-Aware Weight Quantization (AWQ) observes that not all weights are equally important -- a small fraction (1-3%) corresponding to salient activation channels contribute disproportionately to output quality. Rather than quantizing all weights uniformly, AWQ identifies these salient channels by analyzing activation magnitudes on calibration data, then protects them by applying per-channel scaling factors that reduce their relative quantization error: multiply salient weight channels by $s > 1$ before quantization and divide activations by $s$ at runtime. This rebalancing is mathematically equivalent but distributes quantization error away from critical channels. AWQ typically outperforms GPTQ at the same bit-width (especially 4-bit) and is faster to apply since it avoids the column-by-column Hessian computation. It also generalizes better to unseen data since it relies on activation statistics rather than reconstruction on specific calibration examples.

**In Practice:** In MTP, AWQ quantization was evaluated as an alternative to GPTQ for the 8B model serving, ultimately chosen for its superior perplexity-per-bit and faster quantization time during model iteration cycles.

---

### Q69: What is GGUF format?

**Answer:** GGUF (GPT-Generated Unified Format) is a binary file format designed for efficient local inference of quantized models, succeeding the older GGML format. It stores model weights, tokenizer vocabulary, hyperparameters, and metadata in a single self-contained file with a standardized header that enables any compatible runtime (llama.cpp, Ollama, LM Studio) to load and serve the model without external configuration. GGUF supports multiple quantization schemes (Q2_K through Q8_0) within the same format specification, allowing mixed-precision where attention layers retain higher precision (Q6_K) while feed-forward layers use aggressive quantization (Q4_K). The format enables CPU+GPU split inference (offloading $n$ layers to GPU while keeping the rest in RAM), making it accessible on machines with limited VRAM. A typical Q4_K_M quantized 7B model occupies approximately 4.5GB in GGUF format.

**In Practice:** In MTP, GGUF-format models served through Ollama enabled rapid local experimentation on the 0.5B and 8B variants without requiring dedicated GPU servers during the initial model comparison phase.

---

### Q70: How do you estimate token cost for an LLM application?

**Answer:** Token cost estimation requires calculating: (1) input tokens per request = system prompt + conversation history + retrieved context + user message, (2) output tokens per request based on expected response length, (3) requests per time period, then applying the provider's per-token pricing: $\text{Cost} = (T_{\text{in}} \times P_{\text{in}} + T_{\text{out}} \times P_{\text{out}}) \times N_{\text{requests}}$. Output tokens are typically 3-5x more expensive than input tokens (e.g., Claude 3.5 Sonnet: \$3/M input, \$15/M output). Critical multipliers often overlooked include: retry rates (5-15%), conversation turn depth (context grows linearly per turn), RAG chunk overhead (each retrieved passage adds 200-500 tokens), and tool-use round-trips (each tool call doubles the input context). Caching (prompt caching) can reduce costs by 90% for repeated system prompts.

**In Practice:** In SSV RAG, token budgets were managed dynamically via max_output_tokens calibrated by intent (short factual queries got 256 tokens, complex analysis got 4096), reducing monthly costs by approximately 60% compared to uniform max allocation.

---

### Q71: Explain streaming (Server-Sent Events) for LLM responses

**Answer:** Streaming delivers LLM output token-by-token (or in small chunks) to the client as they are generated, using Server-Sent Events (SSE) -- a unidirectional HTTP protocol where the server holds the connection open and sends `data:` prefixed messages. The format follows: `data: {"choices": [{"delta": {"content": "token"}}]}\n\n` with a final `data: [DONE]` signal. This reduces perceived latency from time-to-last-token (TTLT) to time-to-first-token (TTFT), typically 200-500ms regardless of total response length. Implementation requires chunked transfer encoding, proper client-side buffering for partial JSON/markdown rendering, and backpressure handling if the client disconnects mid-stream. For structured output, streaming requires careful buffering -- you cannot validate JSON until the stream completes, requiring either partial parsing or post-stream validation.

**In Practice:** In SSV RAG, streaming was implemented for both Claude and Gemini backends to provide real-time response delivery to analysts, with TTFT under 400ms even for complex retrieval-augmented queries spanning multiple document chunks.

---

### Q72: What causes hallucination and how do you reduce it?

**Answer:** Hallucination occurs when models generate confident but factually incorrect content, caused by: (1) training data contradictions or gaps that leave the model interpolating, (2) the softmax bottleneck forcing probability mass onto tokens even when the model is uncertain, (3) exposure bias from teacher-forcing during training (never seeing its own errors), and (4) the model optimizing for fluency/plausibility rather than factual accuracy. Reduction strategies include: retrieval-augmented generation (grounding in source documents), low temperature ($T \leq 0.2$) for factual tasks, explicit anti-hallucination instructions in the system prompt, structured output with confidence scores to surface uncertainty, and post-generation fact-checking against retrieved evidence. Citation requirements ("quote the source") force the model to ground claims in retrievable evidence.

**In Practice:** In SPOT CHECK, the anti-hallucination principle "Blank > Wrong" was enforced through both prompt engineering and Pydantic validation -- if the model's confidence fell below threshold, the field was returned as null rather than risking a fabricated compliance finding.

---

### Q73: What is the difference between Claude, GPT, Gemini architectures?

**Answer:** While all three are decoder-only transformers using attention mechanisms, they differ in training philosophy and capability emphasis. Claude (Anthropic) emphasizes Constitutional AI alignment with RLAIF, uses large context windows (200K tokens), and excels at instruction following with strong refusal calibration. GPT-4 (OpenAI) pioneered the scaling paradigm with mixture-of-experts architecture (rumored 8x220B), strong code generation, and multimodal vision-language integration. Gemini (Google) is natively multimodal from pre-training (not post-hoc fusion), leverages Google's TPU infrastructure for efficient long-context (up to 2M tokens in Gemini 1.5 Pro), and integrates tightly with Google's search infrastructure for grounding. Key practical differences include context pricing, rate limits, safety filter aggressiveness, and tool-use API design rather than raw capability.

**In Practice:** In SSV RAG, Claude Opus and Gemini 2.5 Pro were deployed as dual backends -- Claude excelled at nuanced regulatory interpretation while Gemini's 2M context window handled full-document ingestion without chunking for simpler queries.

---

### Q74: How do you handle context window overflow?

**Answer:** Context overflow strategies include: (1) sliding window -- drop the oldest messages while preserving the system prompt, (2) summarization -- compress earlier conversation turns into a condensed summary that preserves key facts, (3) RAG-based retrieval -- instead of retaining full history, embed and retrieve only relevant past turns, (4) hierarchical memory -- maintain a short-term buffer (recent turns) plus long-term store (summarized or embedded history), and (5) token-aware truncation -- compute token counts incrementally and truncate at semantic boundaries (paragraph/sentence level) when approaching limits. The formula for available user tokens is: $T_{\text{available}} = T_{\text{window}} - T_{\text{system}} - T_{\text{output\_reserved}} - T_{\text{tools}}$. Priority ordering matters: system prompt and tool definitions are non-negotiable, recent user message is essential, then conversation history fills remaining space from most recent backward.

**In Practice:** In iFAST, context overflow was managed by maintaining a sliding window of the last 10 turns plus a running summary of earlier conversation, ensuring the 14-tool system prompt always had sufficient remaining space for user queries and tool responses.

---

### Q75: What is prompt injection and how to prevent it?

**Answer:** Prompt injection is an adversarial attack where user input manipulates the model into ignoring its system prompt instructions -- either directly ("ignore previous instructions and...") or indirectly (embedding malicious instructions in retrieved documents or tool outputs). Prevention layers include: (1) input sanitization (detecting and filtering injection patterns), (2) delimiter-based separation (clearly marking boundaries between system/user/retrieved content using XML tags or special tokens), (3) output validation (checking model responses against expected schema/constraints), (4) privilege separation (the model cannot perform dangerous actions regardless of instructions), and (5) canary tokens (embedding detectable strings in system prompts to verify they haven't been overridden). No single defense is sufficient; defense-in-depth combining multiple layers is required, and the problem remains fundamentally unsolved since the model processes instructions and data in the same channel.

**In Practice:** In iFAST, prompt injection defenses included XML-delimited content boundaries, Pydantic validation of all tool call outputs, and a secondary Claude call that verified tool selection plausibility before execution -- preventing users from manipulating the model into unauthorized data access.

---

### Q76: Explain model distillation for LLMs

**Answer:** Model distillation trains a smaller "student" model to replicate the behavior of a larger "teacher" model by learning from the teacher's soft probability distributions rather than hard labels: $\mathcal{L} = \alpha \cdot \text{KL}(P_T(x; T) \| P_S(x; T)) + (1-\alpha) \cdot \mathcal{L}_{\text{CE}}$, where $T$ is a temperature that softens distributions and $\alpha$ balances distillation loss against standard cross-entropy. The soft targets carry "dark knowledge" -- information about similarity structure between classes that hard labels discard (e.g., the teacher assigns 0.3 to a synonym vs 0.001 to an unrelated word). For LLMs, distillation often takes the form of training on teacher-generated outputs (synthetic data distillation) rather than matching full logit distributions, since accessing logits of proprietary models is infeasible. This is how many open-source models (Alpaca, Vicuna) achieved instruction-following ability -- by training on GPT-4 outputs.

**In Practice:** In MTP, the comparison between 0.5B, 8B, and 14B models implicitly evaluated distillation trade-offs -- the 8B model (likely distilled from a larger teacher) achieved 85% of the 14B's branching quality at 40% of the inference cost.

---

### Q77: What is Constitutional AI (RLHF vs RLAIF)?

**Answer:** Constitutional AI (CAI) replaces human preference labeling (RLHF) with AI-generated feedback (RLAIF): a set of written principles ("the constitution") guides a model to critique and revise its own outputs, generating preference pairs that train a reward model without human annotators. In RLHF: $\text{reward model} \leftarrow \text{human comparisons} \to \text{PPO fine-tuning}$. In RLAIF: $\text{reward model} \leftarrow \text{AI-judged comparisons via constitution} \to \text{PPO fine-tuning}$. The constitutional approach scales better (no human bottleneck), provides more consistent judgments, and makes the alignment criteria transparent and auditable. However, it inherits biases of the AI judge and requires careful constitution design to avoid reward hacking. Claude's training combines both: RLHF for initial alignment plus CAI principles for scalable refinement of edge cases.

**In Practice:** In iFAST, understanding Constitutional AI informed the design of anti-hallucination system prompts -- the principle of explicit behavioral constraints mirrors how constitutional principles guide Claude's own training.

---

### Q78: How do you evaluate LLM output quality?

**Answer:** LLM evaluation spans automatic metrics and human/model judgment: (1) reference-based metrics (BLEU, ROUGE, BERTScore) compare against gold answers but correlate poorly with human judgment for open-ended generation, (2) LLM-as-judge uses a stronger model to rate outputs on criteria (helpfulness, accuracy, harmlessness) with rubrics, achieving 80-90% agreement with human raters, (3) task-specific metrics (exact match for QA, pass@k for code, factual precision/recall for RAG), and (4) A/B testing with real users measuring engagement and task completion. The evaluation framework should match the deployment context: $\text{Quality} = w_1 \cdot \text{Accuracy} + w_2 \cdot \text{Completeness} + w_3 \cdot \text{Format Compliance} + w_4 \cdot \text{Safety}$, with weights tuned per use case. Regression testing on curated eval sets is essential before deploying prompt or model changes.

**In Practice:** In SPOT CHECK, LLM output quality was evaluated by comparing model compliance assessments against expert-labeled ground truth, measuring precision/recall per violation category and tracking the "Blank > Wrong" null-rate as a calibration signal.

---

### Q79: What is the difference between base models and instruction-tuned?

**Answer:** Base models are trained purely on next-token prediction over large text corpora, learning language patterns, world knowledge, and reasoning capabilities but producing completions that continue the statistical patterns of their training data rather than following user instructions. Instruction-tuned models undergo additional training phases: supervised fine-tuning (SFT) on instruction-response pairs (thousands to millions of examples) followed by RLHF/RLAIF alignment to learn human-preferred response styles. The base model might complete "What is 2+2?" with "What is 2+3? What is 3+3?..." (continuing a pattern), while the instruction-tuned version answers "4." Base models excel at few-shot in-context learning and domain adaptation via continued pre-training, while instruction-tuned models are ready for chat deployment but can be harder to steer for unconventional formats. The alignment tax (slight capability reduction from RLHF) is typically 1-3% on benchmarks but yields dramatically more usable outputs.

**In Practice:** In MTP, instruction-tuned Qwen models were used for the branching generation since they reliably followed structured output instructions, whereas base models would have required extensive few-shot formatting examples consuming precious context space.

---

### Q80: How do you choose between model sizes (0.5B vs 8B vs 14B vs 70B)?

**Answer:** Model size selection balances capability against latency, cost, and infrastructure constraints using the principle of "smallest model that meets quality threshold." The decision framework considers: (1) task complexity -- classification/extraction tasks often work at 8B while multi-step reasoning needs 70B+, (2) latency budget -- 0.5B generates at 200+ tokens/sec on consumer GPUs vs 20 tok/sec for 70B on A100, (3) cost at scale -- 10x parameters means roughly 10x compute cost per token, (4) deployment constraints -- edge/mobile requires sub-3B, single GPU allows up to Q4-70B (~40GB), and (5) fine-tuning feasibility -- smaller models are cheaper to adapt to specific domains. The empirical approach is to benchmark on representative evaluation sets across sizes and pick the knee of the quality-cost curve: $\text{optimal size} = \arg\min_{s} \text{Cost}(s) \text{ s.t. } \text{Quality}(s) \geq \tau$.

**In Practice:** In MTP, systematic comparison of 0.5B vs 8B vs 14B Qwen models revealed that 0.5B was insufficient for coherent multi-step reasoning, 14B offered marginal gains over 8B for 75% more compute, making the 8B model the cost-optimal choice for production branching at $T=0.45$.
