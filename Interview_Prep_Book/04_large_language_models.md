# Chapter 4: Large Language Models — How They Work

*From tokenization to inference optimization — the mechanics behind every LLM-powered system in this book.*

---

Large Language Models are the generation engine in every system described in this book. The SSV RAG chatbot retrieves context, but it is an LLM that synthesizes the answer. iFAST orchestrates 14 tools, but it is an LLM that decides which tool to call. SPOT CHECK enforces structured evaluation, but it is an LLM that produces the JSON. Understanding how these models work — not as magic, but as machinery — is what separates engineers who build production AI systems from engineers who call APIs and hope.

This chapter covers the full stack: tokenization, autoregressive generation, sampling, context windows, KV-caching, model families, prompt engineering, tool-use, structured output, and inference optimization. Each concept connects back to a real system.

---

## 4.1 What Is a Large Language Model?

A Large Language Model is a neural network trained on massive text corpora to predict the next token in a sequence. That is the entire training objective — next-token prediction — yet from this single objective, capabilities emerge: reasoning, code generation, summarization, translation, tool use, and more.

**Scale defines "large":**
- **Parameters:** 0.5B (Qwen-0.5B) to 405B (LLaMA-3.1-405B). Each parameter is a learned weight in the network.
- **Training data:** Trillions of tokens. LLaMA 3 was trained on 15T tokens. GPT-4's training data is undisclosed but estimated at 10-13T tokens.
- **Compute:** Training a 70B model costs roughly $2-5M in GPU-hours.

**Emergent capabilities:** Certain abilities appear only above a critical scale threshold. A 1B model cannot reliably do multi-step arithmetic. A 7B model can. A 7B model cannot reliably use tools with complex schemas. A 70B model can. This is not a smooth improvement — it is a phase transition. My MTP thesis studied exactly this: comparing 0.5B, 8B, and 14B models on structured reasoning tasks, observing where each model's capabilities hit their ceiling.

---

## 4.2 Tokenization

Neural networks operate on numbers, not text. Tokenization converts raw text into a sequence of integer IDs that the model can process.

### Byte-Pair Encoding (BPE)

The dominant tokenization algorithm. The procedure:

1. Start with the raw text split into individual characters (or bytes)
2. Count all adjacent character pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat steps 2-3 for a fixed number of merge operations (determines vocabulary size)

**Example:** Given the corpus "low lower lowest", BPE might merge:
- `l` + `o` → `lo` (most frequent pair)
- `lo` + `w` → `low` (next most frequent)
- `e` + `r` → `er`
- `e` + `s` → `es`

After training, "lowest" tokenizes as `["low", "es", "t"]`.

### SentencePiece

Google's language-agnostic tokenizer. Unlike BPE which operates on pre-tokenized words, SentencePiece treats the input as a raw byte stream — no language-specific preprocessing needed. It uses either BPE or Unigram algorithms internally but operates directly on Unicode text, making it suitable for multilingual models.

### Vocabulary Size Tradeoffs

| Vocab Size | Tokens per Text | Parameters | Use Case |
|-----------|----------------|------------|----------|
| Small (8K) | More tokens per sentence | Fewer embedding params | Byte-level models |
| Medium (32K) | Balanced | Balanced | LLaMA, most open models |
| Large (100K+) | Fewer tokens per sentence | More embedding params | GPT-4, Claude |

Larger vocabularies mean each token carries more information (fewer tokens per sentence = faster inference, lower cost), but the embedding table grows proportionally.

### Token is Not Word

This is a common source of confusion:
- "unhappiness" might tokenize as `["un", "happiness"]` or `["unhapp", "iness"]`
- "ChatGPT" might be `["Chat", "G", "PT"]`
- Code: `print("hello")` might be `["print", "(\"", "hello", "\")"]`
- Numbers: "123456" is often split character by character

**Why this matters practically:** When you set `max_output_tokens=500`, that is 500 *tokens*, not 500 words. English averages roughly 0.75 words per token (or ~4 characters per token). A 4096-token context window holds approximately 3000 words.

---

## 4.3 Autoregressive Generation

LLMs generate text one token at a time, left to right. Each token is predicted conditioned on all previous tokens:

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

### Training: Teacher Forcing

During training, the model sees the correct previous tokens (from the training data) and predicts the next one. The loss is cross-entropy between the predicted probability distribution and the actual next token. This is called "teacher forcing" because the true sequence "teaches" the model, regardless of what the model would have predicted.

### Inference: Sequential Generation

At inference time, there is no teacher. The model:
1. Takes the input prompt as the initial sequence
2. Predicts a probability distribution over the vocabulary for the next token
3. Samples (or greedily selects) one token from that distribution
4. Appends the selected token to the sequence
5. Repeats from step 2 until a stop condition (EOS token, max length)

### Why Inference Is Slow

Each token generation step requires a forward pass through the entire model. For a 70B parameter model generating 500 tokens, that is 500 sequential forward passes — each one depending on the output of the previous. This is inherently serial: you cannot predict token 5 until you know token 4.

This is why LLM inference is fundamentally different from LLM training. Training is embarrassingly parallel (all tokens in the training sequence are predicted simultaneously against ground truth). Inference is $O(n)$ sequential steps, where $n$ is the output length.

---

## 4.4 Sampling Strategies

After the model computes logits (raw scores) for each token in the vocabulary, we must decide which token to actually select. This decision dramatically affects output quality.

### Greedy Decoding (Temperature = 0)

Always select the highest-probability token. Deterministic — same input always produces same output. Produces coherent but repetitive, "safe" text. Used when you need reproducibility (testing, structured output).

### Temperature Scaling

Apply a temperature parameter $T$ before the softmax:

$$P(x_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

Where $z_i$ is the raw logit for token $i$.

- **T < 1:** Distribution becomes sharper. High-probability tokens get even higher probability. More deterministic. Good for factual tasks.
- **T = 1:** Original distribution unchanged.
- **T > 1:** Distribution becomes flatter. Low-probability tokens become more likely. More creative/random. Good for brainstorming, creative writing.

### Top-k Sampling

Only consider the $k$ highest-probability tokens. Zero out everything else, renormalize. Problem: the optimal $k$ varies by context. After "The capital of France is", only 1-2 tokens are reasonable. After "The color of the dress was", many tokens are reasonable.

### Top-p (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds $p$. Adapts to the shape of the distribution — narrow distributions (confident model) consider few tokens; flat distributions (uncertain model) consider many.

**Typical production settings:**
- Factual Q&A: T=0.0 (greedy) or T=0.1, top_p=0.9
- Creative generation: T=0.7-1.0, top_p=0.95
- Code generation: T=0.0 (deterministic, correct code matters)

### Why This Matters for Search

In my MTP thesis, we use varied temperatures to produce diverse reasoning paths for A* search. Temperature=0 gives one deterministic path. Temperature=0.7 gives varied alternatives. The search algorithm then selects the best path across these candidates — higher temperature creates the diversity that makes search worthwhile.

---

## 4.5 Context Window & KV-Cache

### Context Window

The context window is the maximum number of tokens the model can process in a single forward pass. It is a hard architectural limit determined during training.

| Model | Context Window |
|-------|---------------|
| LLaMA 3 (8B) | 8,192 tokens |
| LLaMA 3.1 (70B) | 128K tokens |
| Claude 3.5 Sonnet | 200K tokens |
| GPT-4 Turbo | 128K tokens |
| Gemini 1.5 Pro | 1M tokens |

Everything beyond the context window is invisible to the model. This is why RAG systems must carefully manage what goes into context — sending irrelevant chunks wastes precious tokens and can confuse the model.

### Attention Complexity

Self-attention is $O(n^2)$ in sequence length: every token attends to every other token. Doubling context length quadruples the compute cost of the attention layers. This is why longer context windows are expensive and why models with 1M token contexts require architectural innovations (sparse attention, linear attention approximations).

### KV-Cache: The Critical Optimization

During autoregressive generation, each new token must attend to all previous tokens. Naively, this means recomputing the Key and Value projections for all previous tokens at every step — $O(n^2)$ total work for generating $n$ tokens.

**The KV-Cache stores the computed Key and Value matrices for all previously generated tokens.** At each step, only the new token's Query, Key, and Value need to be computed. The attention for the new token is then:

$$\text{Attention}(q_\text{new}, K_\text{cached}, V_\text{cached}) = \text{softmax}\left(\frac{q_\text{new} \cdot K_\text{cached}^T}{\sqrt{d_k}}\right) V_\text{cached}$$

**Without KV-Cache:** Each of the $n$ generation steps recomputes attention over all previous tokens → $O(n^2)$ total.

**With KV-Cache:** Each step only computes attention for the new token against cached KV → $O(n)$ per step, $O(n^2)$ total memory but $O(n)$ compute per step.

**Memory cost:** KV-cache size = $2 \times \text{layers} \times \text{heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{precision}$

For LLaMA 70B with 80 layers, 64 heads, head_dim=128, and FP16 precision:
- Per token: $2 \times 80 \times 64 \times 128 \times 2$ bytes = 2.6 MB
- For 4096 tokens: ~10.5 GB of GPU memory *just for KV-cache*

This is why inference serving is a memory problem, not just a compute problem. It is also why vLLM's PagedAttention (Section 4.10) matters so much.

---

## 4.6 Model Families

| Family | Sizes | Open/Closed | Key Feature |
|--------|-------|-------------|-------------|
| **LLaMA (Meta)** | 8B-405B | Open weights | Strong open-source baseline, extensive fine-tuning ecosystem |
| **Qwen (Alibaba)** | 0.5B-72B | Open weights | Competitive at small scale, good multilingual |
| **Claude (Anthropic)** | Undisclosed | Closed API | Long context (200K), strong tool-use, steerable |
| **GPT (OpenAI)** | Undisclosed | Closed API | Multimodal, function calling, ecosystem |
| **Gemini (Google)** | Undisclosed | Closed API | Multimodal, 1M context, grounded search |
| **Mistral** | 7B-8x22B | Open weights | Mixture-of-Experts (MoE), efficient for size |

**Open vs Closed tradeoffs:**
- Open weights: self-host, fine-tune, no per-token cost, full control. But: you pay for GPU infrastructure, handle scaling, deal with quantization.
- Closed API: immediate access to frontier capability, managed infrastructure. But: per-token cost, vendor lock-in, data leaves your premises.

**Production reality:** The SSV RAG chatbot uses Claude (primary) with Gemini fallback — closed APIs chosen because corporate policy demanded managed infrastructure. My MTP research uses open models (Qwen 0.5B/7B/14B, LLaMA 8B) because we needed to control sampling, inspect internals, and run thousands of inference calls without API costs.

---

## 4.7 Prompt Engineering

Prompt engineering is the practice of crafting inputs that reliably elicit desired outputs from LLMs. It is not "tricks" — it is the interface design between humans and language models.

### System Prompt

Sets the model's role, constraints, and output format. Persists across the conversation.

```
You are a diagnostic AI that identifies root causes of software failures.
Rules:
1. Only cite evidence from the provided tool outputs.
2. Express confidence as a probability between 0 and 1.
3. If evidence is insufficient, say so explicitly.
Output format: JSON with keys "root_cause", "confidence", "evidence_chain".
```

### Few-Shot Prompting

Include 2-5 examples of desired input-output behavior:

```
Query: "Test failed with timeout on CAN bus"
Answer: {"root_cause": "CAN bus hardware timeout", "confidence": 0.82, ...}

Query: "NullPointerException in module X"
Answer: {"root_cause": "Uninitialized dependency injection", "confidence": 0.71, ...}

Query: [ACTUAL USER QUERY]
Answer:
```

### Chain-of-Thought (CoT)

Explicitly instruct the model to reason step-by-step before answering:

```
Think step by step:
1. Identify the failing component from the error log
2. List possible root causes for this failure mode
3. Check which causes are supported by the provided evidence
4. Select the most likely cause and state your confidence
```

CoT improves performance on multi-step reasoning tasks. It is most effective for models above 7B parameters — smaller models often produce incoherent reasoning chains that hurt rather than help.

### Anti-Hallucination Prompting

```
IMPORTANT: Only use information explicitly present in the retrieved context below.
If the answer is not contained in the context, respond with: "I don't have enough information to answer this question."
Do NOT infer, guess, or use information from your training data.
```

The SSV RAG chatbot uses this pattern. Combined with source citation requirements, it reduces hallucination significantly — our 100-query benchmark showed 100% guardrail compliance on out-of-distribution queries.

---

## 4.8 Tool-Use & Function Calling

Tool-use is the mechanism by which LLMs interact with external systems. Instead of generating text, the model outputs a structured function call that the orchestrating system executes.

### How It Works

1. Define available tools with JSON Schema (name, description, parameters)
2. Send user query + tool definitions to the model
3. Model decides: respond with text OR output a tool call
4. System executes the tool call, returns result to model
5. Model generates final response incorporating tool output

### Tool Definition Example (from iFAST)

```json
{
  "name": "search_jira_issues",
  "description": "Search Jira issues using JQL. Use for finding bugs, tasks, or stories matching specific criteria.",
  "parameters": {
    "type": "object",
    "properties": {
      "jql_query": {"type": "string", "description": "Valid JQL query string"},
      "max_results": {"type": "integer", "default": 20, "maximum": 100}
    },
    "required": ["jql_query"]
  }
}
```

### Parallel Tool Calls

Advanced models can call multiple tools simultaneously when the calls are independent:

```json
[
  {"tool": "search_jira_issues", "args": {"jql_query": "project=EDS AND type=Bug"}},
  {"tool": "get_confluence_page", "args": {"page_id": "12345"}}
]
```

iFAST uses this extensively — when diagnosing a failure, the agent often needs data from multiple sources simultaneously. Parallel calls reduce latency from sequential round-trips.

### Error Handling

If a tool returns an error, the model can:
- Retry with corrected parameters (e.g., fix a malformed JQL query)
- Try a different tool that might provide similar information
- Report to the user that the required data is unavailable

iFAST implements all three patterns. The 14-tool orchestration handles tool failures gracefully because Claude's tool-use behavior includes built-in retry logic — it observes the error message and adapts.

---

## 4.9 Structured Output

LLMs produce free text by default, but production systems need structured data — JSON, enums, typed fields.

### The Problem

Ask an LLM to "output JSON" and you might get:
- Valid JSON wrapped in markdown code fences
- JSON with trailing commas (invalid)
- A conversational preamble before the JSON
- Missing required fields

### JSON Mode

Some APIs offer a JSON mode that constrains the model to output syntactically valid JSON. This guarantees parseable output but does NOT guarantee the schema is correct.

### Pydantic Validation (Used in SPOT CHECK)

Define the expected schema, validate LLM output, retry on failure:

```python
from pydantic import BaseModel, Field
from typing import Literal

class AuditResult(BaseModel):
    criterion: str = Field(description="The quality criterion being assessed")
    score: Literal["Pass", "Fail", "Partial"]
    evidence: str = Field(min_length=20, description="Direct quote from the document")
    confidence: float = Field(ge=0.0, le=1.0)

# Parse LLM output
try:
    result = AuditResult.model_validate_json(llm_output)
except ValidationError as e:
    # Retry with error message appended to prompt
    retry_prompt = f"{original_prompt}\n\nYour previous output had errors: {e}\nPlease fix and try again."
```

SPOT CHECK V2 uses this pattern for every evaluation criterion. The dual-agent system produces structured assessments with mandatory evidence citation — Pydantic enforces that the evidence field is never empty and the score is always one of the valid options.

### Guided Generation

The most aggressive approach: constrain token-by-token generation so that only tokens keeping the output JSON-valid are allowed. Libraries like Outlines and Guidance implement this by masking logits at each generation step. This guarantees schema-valid output on the first attempt with no retries.

---

## 4.10 Inference Optimization

Running LLMs in production is expensive. These techniques reduce cost and latency.

### vLLM and PagedAttention

The key insight: KV-cache memory is wasted due to fragmentation (reserved but unused space for sequences that end early). PagedAttention manages KV-cache like OS virtual memory — allocating in fixed-size "pages" and mapping them to physical GPU memory on demand.

- **Continuous batching:** Instead of waiting for all sequences in a batch to finish, immediately fill completed slots with new requests. Increases throughput 2-4x.
- **Prefix caching:** Shared system prompts are cached once and reused across requests.

### Quantization

Reduce model precision from FP16 (2 bytes/param) to INT4 (0.5 bytes/param):

| Method | Precision | Speed | Quality Loss | Use Case |
|--------|-----------|-------|-------------|----------|
| **GPTQ** | 4-bit | Fast | Low (~1-2% on benchmarks) | GPU inference |
| **AWQ** | 4-bit | Fast | Lower than GPTQ | GPU inference (better quality) |
| **GGUF** | 2-8 bit | Variable | Depends on bits | CPU inference (llama.cpp, Ollama) |

A 70B model in FP16 requires 140GB of GPU memory. In 4-bit quantization, it fits in 35GB — a single A100. This is how open models become practical for deployment.

### Ollama

Local model serving for development and small-scale deployment:
- Automatic quantization and model management
- Single command: `ollama run llama3:8b`
- REST API compatible with OpenAI SDK format
- Suitable for development, testing, and edge deployment

### Speculative Decoding

Use a small "draft" model (e.g., 1B parameters) to generate $k$ candidate tokens quickly. The large "verifier" model (e.g., 70B) checks all $k$ tokens in parallel (a single forward pass). Accepted tokens are kept; rejected tokens are regenerated by the large model.

Speedup: 2-3x for tasks where the small model predicts correctly most of the time (simple tokens like common words, punctuation). No quality loss — the output distribution is mathematically identical to the large model alone.

### Batching

Process multiple user requests together in a single forward pass. GPU utilization jumps from 10-20% (single request) to 80-90% (batched). The key insight: attention computation parallelizes across sequences in a batch, but not across tokens within a sequence.

---

## 4.11 Cost & Token Management

LLM APIs charge per token. Without active management, costs spiral.

### Pricing Structure

Typical pricing (mid-2025):
- Input tokens: $3-15 per million tokens (Claude Sonnet: $3/M input)
- Output tokens: $15-75 per million tokens (Claude Sonnet: $15/M output)
- Output is 3-5x more expensive because it requires sequential generation

### Token Budgeting

Set `max_output_tokens` per intent type:

```python
TOKEN_BUDGETS = {
    "count_query": 256,      # "How many open bugs?" → short answer
    "summary_query": 1024,   # "Summarize this sprint" → medium
    "analysis_query": 4096,  # "Analyze root cause" → detailed
    "meta_query": 128,       # "What can you do?" → very short
}
```

The SSV RAG chatbot classifies intent first, then sets the appropriate token budget. A count query ("How many bugs does EDS have?") does not need 8192 output tokens — 256 is sufficient and costs 32x less.

### Context Compression

Before sending to the LLM, compress the retrieved context:
- Remove duplicate chunks (often retrieved by both BM25 and ANN)
- Truncate long chunks to the most relevant paragraphs
- Enforce a hard character limit (SSV RAG uses 15,000 characters)
- Order by relevance score (most relevant first — if context is truncated, the best chunks survive)

### Response Caching

Identical prompts should not call the LLM twice:

```python
cache_key = hashlib.md5(prompt.encode()).hexdigest()
if cache_key in disk_cache:
    return disk_cache[cache_key]  # Skip LLM entirely
```

SSV RAG's semantic cache goes further — queries that are *similar* (cosine similarity > 0.92) also hit cache. This covers paraphrases: "Show me EDS bugs" and "What are the bugs for EDS?" share a cached response.

---

## 4.12 Summary & Interview Tips

### Must-Know Concepts

| Concept | One-Sentence Explanation |
|---------|--------------------------|
| Tokenization | Text → integer IDs via BPE; token is not word |
| Autoregressive | Generate one token at a time conditioned on all previous |
| Temperature | Controls randomness: T=0 deterministic, T>1 more random |
| KV-Cache | Store computed Keys/Values to avoid recomputing at each step |
| Tool-use | Model outputs structured function calls instead of text |
| Structured output | Constrain LLM to produce valid JSON matching a schema |
| Quantization | Reduce precision (FP16→INT4) to fit large models on less hardware |

### Common Interview Questions

**"Why is LLM inference slow?"**
Autoregressive generation is inherently sequential — each token depends on the previous. You cannot parallelize token generation within a single sequence. For $n$ output tokens, you need $n$ sequential forward passes through a model with billions of parameters.

**"How do you reduce LLM costs?"**
1. Cache identical/similar prompts (semantic cache)
2. Set per-intent token budgets (don't generate 4K tokens for a yes/no question)
3. Use smaller models for simpler tasks (route easy queries to 8B, hard queries to 70B)
4. Compress context before sending (remove irrelevant retrieved chunks)
5. Batch requests for better GPU utilization

**"Explain KV-cache to me."**
During generation, each new token attends to all previous tokens. Without caching, you recompute Key and Value projections for all tokens at every step. KV-cache stores these projections — at each step, only the new token's projections are computed. This reduces per-step compute from $O(n)$ to $O(1)$ (for the K/V computation, not the attention itself).

**"What is the difference between fine-tuning and prompting?"**
Prompting adapts behavior at inference time (zero cost, instant, limited by context window). Fine-tuning adapts model weights permanently (expensive, time-consuming, but the model *internalizes* the desired behavior). Use prompting first; fine-tune only when prompting cannot achieve the required quality or when per-token costs of long system prompts exceed fine-tuning amortized cost.

---

## 4.13 Connections to Author's Projects

Every section of this chapter maps to a real system:

**MTP Thesis (A* + Topology Router):** Compared Qwen 0.5B, 7B, and 14B on structured reasoning. Observed scaling laws in practice — the 14B model solved problems the 0.5B model could not, validating emergent capability thresholds. Used varied temperatures (Section 4.4) to generate diverse reasoning paths for search.

**iFAST (Mercedes-Benz):** Production system with 14 tools orchestrated by Claude. Demonstrates tool-use at scale (Section 4.8) — the model selects tools, handles errors, makes parallel calls, and synthesizes multi-source evidence into structured diagnoses.

**SSV RAG Chatbot:** Manages dual LLM backends (Claude + Gemini) with streaming, token budgets per intent type, context compression to 15K characters, and three-tier caching. Every concept from Sections 4.5, 4.11 is directly implemented.

**SPOT CHECK V2:** Dual-agent compliance system that enforces structured JSON output via Pydantic schemas (Section 4.9). Every assessment must include a valid score, mandatory evidence citation, and confidence value — all validated programmatically before being accepted.

These are not academic exercises. They are production systems serving real users, processing real data, under real latency and cost constraints.
