# Chapter 3: Deep Learning & Neural Networks

*From perceptrons to transformers to LoRA — the architecture evolution that made LLMs possible, and the training techniques that make fine-tuning practical.*

---

Deep learning is the foundation on which every system in this book is built. The SSV RAG chatbot calls Claude — a transformer. iFAST orchestrates Claude for tool-use — a transformer with function calling. The MTP thesis fine-tunes LLaMA with LoRA — a transformer adapted with parameter-efficient methods. SPOT CHECK's dual agents are both transformers producing structured output.

You do not need to train a transformer from scratch to be effective as an AI engineer. But you need to understand how they work — deeply enough that when something goes wrong (generation degrades, fine-tuning diverges, latency spikes), you can reason about why and fix it. This chapter builds that understanding from the ground up: basic neural networks, training challenges, convolutional and recurrent architectures (for historical context), then the transformer architecture that dominates today, and finally the parameter-efficient fine-tuning methods that make adaptation practical.

---

## 3.1 Neural Network Basics

### The Perceptron

The simplest neural network: a single neuron. It computes a weighted sum of inputs, adds a bias, and applies an activation function:

$$y = \sigma(w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

Where $\sigma$ is a nonlinear activation function. Without $\sigma$, the network is just a linear transformation — stacking linear layers produces another linear layer. The activation function introduces nonlinearity, which gives neural networks their expressive power.

### Multi-Layer Networks

Stack multiple layers of neurons. The output of one layer becomes the input of the next.

**Universal Approximation Theorem:** A feedforward network with a single hidden layer of sufficient width can approximate any continuous function on a compact set to arbitrary precision. This is a theoretical existence result — it does not tell you how to find the right weights, and "sufficient width" may be impractically large. In practice, depth (many layers) is more parameter-efficient than width (one huge layer).

### Activation Functions

**ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$. Simple, fast, and effective. Solves the vanishing gradient problem for positive inputs. Problem: "dead neurons" — if a neuron's input is always negative, its gradient is always zero and it never updates.

**Sigmoid:** $f(x) = \frac{1}{1+e^{-x}}$. Squashes output to (0,1). Useful for binary outputs (probabilities). Problem: gradients vanish for large or small inputs (saturation). Largely replaced by ReLU in hidden layers.

**Tanh:** $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$. Squashes to (-1, 1). Zero-centered (unlike sigmoid), which helps optimization. Same saturation problem as sigmoid.

**GELU (Gaussian Error Linear Unit):** $f(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard Gaussian CDF. Smooth approximation of ReLU. Used in BERT, GPT, and most modern transformers. Slightly better empirical performance than ReLU for language tasks.

### Loss and Backpropagation

**Forward pass:** Input flows through the network layer by layer, producing an output. The loss function computes how wrong the output is.

**Backward pass (backpropagation):** The gradient of the loss flows backward through the network using the chain rule of calculus. Each layer receives the gradient from above and computes (a) the gradient with respect to its own parameters (used for weight updates) and (b) the gradient with respect to its input (passed to the layer below).

**Computational graph:** The network is a directed acyclic graph of operations. Each operation (matrix multiply, activation, addition) has a local gradient that can be computed efficiently. Backpropagation applies the chain rule by multiplying these local gradients along every path from the loss to each parameter.

This is not magic. It is calculus applied systematically through a graph. Frameworks (PyTorch, TensorFlow) automate this — you define the forward pass and the framework computes gradients automatically.

---

## 3.2 Training Challenges

### Vanishing Gradients

In deep networks, gradients pass through many layers during backpropagation. If each layer multiplies the gradient by a factor less than 1 (as sigmoid and tanh do in their saturated regions), the gradient shrinks exponentially. Layers far from the output receive near-zero gradients and do not learn.

**Solutions:**
- **ReLU activation:** Gradient is 1 for positive inputs — no shrinkage.
- **Residual connections (ResNets):** Add skip connections that let gradients flow directly to earlier layers, bypassing intermediate layers entirely. The gradient for a residual block is $1 + \frac{\partial f}{\partial x}$ rather than just $\frac{\partial f}{\partial x}$ — the "1" ensures a gradient always flows through.
- **Careful initialization:** Xavier/Glorot initialization sets weights so that activation variance remains constant across layers.

### Exploding Gradients

The opposite problem: gradients grow exponentially through layers. Weights update by enormous amounts, causing parameters to diverge to infinity.

**Solution — Gradient Clipping:** If the gradient norm exceeds a threshold $g_{max}$, scale it down:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{g_{max}}{||\mathbf{g}||}$$

This caps the maximum update size without changing the gradient direction. Standard practice in transformer training — typical max norm is 1.0.

### Batch Normalization

Normalize the activations of each layer across the mini-batch:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

Then apply learnable scale ($\gamma$) and shift ($\beta$) parameters. This stabilizes training by keeping activations in a well-behaved range, allows higher learning rates, and acts as a mild regularizer. Used extensively in CNNs; transformers use Layer Normalization (normalize across features rather than across the batch) because it works better for variable-length sequences.

### Dropout

During training, randomly set each neuron's output to zero with probability $p$ (typically 0.1-0.5). At inference time, scale outputs by $(1-p)$ to compensate.

**Why it works:** Forces the network to be redundant — no single neuron can be relied upon. The network learns distributed representations that are robust to individual neuron failures. Equivalent to training an exponential number of "thinned" networks and averaging their predictions (ensemble interpretation).

---

## 3.3 Convolutional Neural Networks (CNNs)

While not directly used in the author's projects (which are NLP-focused), CNNs are foundational knowledge for interviews and illustrate key principles that reappear in transformers.

### Convolution

A convolutional layer applies a small learned filter (e.g., 3x3) across the entire input. The same filter is reused at every spatial location.

**Three key properties:**
1. **Local receptive fields:** Each output neuron depends only on a small region of the input. Captures local patterns (edges, textures).
2. **Weight sharing:** The same filter weights are used everywhere. Drastically reduces parameters compared to fully connected layers.
3. **Translation invariance:** A pattern detected in one location is detected everywhere. A cat in the top-left corner activates the same feature as a cat in the bottom-right.

### Pooling

Reduces spatial dimensions by aggregating neighboring values:
- **Max pooling:** Take the maximum value in each region. Retains the strongest activation.
- **Average pooling:** Take the mean. Smoother but may lose strong signals.

Pooling provides translation invariance and reduces computation for subsequent layers.

### Typical Architecture

$$\text{Input} \rightarrow [\text{Conv} \rightarrow \text{ReLU} \rightarrow \text{Pool}] \times N \rightarrow \text{Flatten} \rightarrow \text{FC} \rightarrow \text{Output}$$

Early layers capture low-level features (edges, corners). Middle layers capture mid-level features (textures, parts). Final layers capture high-level features (objects, scenes). This hierarchical feature extraction is what makes CNNs powerful for vision tasks.

---

## 3.4 Recurrent Neural Networks

RNNs process sequences by maintaining a hidden state that carries information from previous time steps to future ones.

### Basic RNN

At each time step $t$:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

The hidden state $h_t$ encodes the "memory" of everything seen so far. Problem: in practice, this memory is short. The vanishing gradient problem is severe in RNNs — gradients flow through the same weight matrix $W_h$ at every time step, and the product of many matrices either vanishes or explodes.

### LSTM (Long Short-Term Memory)

Introduces a cell state $c_t$ and three gates that control information flow:

- **Forget gate:** $f_t = \sigma(W_f [h_{t-1}, x_t])$ — what to discard from the cell state
- **Input gate:** $i_t = \sigma(W_i [h_{t-1}, x_t])$ — what new information to store
- **Output gate:** $o_t = \sigma(W_o [h_{t-1}, x_t])$ — what to output from the cell state

The cell state update: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$

The forget gate can learn to pass information unchanged across many time steps (by outputting values near 1), solving the vanishing gradient problem for long-range dependencies.

### GRU (Gated Recurrent Unit)

Simplified LSTM with two gates instead of three:
- **Reset gate:** Controls how much of the previous hidden state to forget
- **Update gate:** Controls the mix of previous hidden state and new candidate

Fewer parameters than LSTM, comparable performance on many tasks.

### Why RNNs Lost

RNNs process sequences token by token — inherently sequential. Training cannot be parallelized across time steps. For a sequence of length $n$, you need $n$ sequential operations. Transformers process all tokens in parallel (attention sees everything at once), making training orders of magnitude faster on modern GPUs.

---

## 3.5 The Transformer Architecture

The transformer (Vaswani et al., 2017, "Attention Is All You Need") replaced RNNs for virtually all NLP tasks and is the foundation of every modern LLM. Understanding this architecture is non-negotiable for AI interviews.

### Self-Attention Mechanism

The core operation. Given a sequence of token representations, self-attention allows each token to "look at" every other token and compute a weighted combination based on relevance.

**Step 1:** Project each token into three vectors using learned matrices:
- Query ($Q = XW_Q$): "What am I looking for?"
- Key ($K = XW_K$): "What do I contain?"
- Value ($V = XW_V$): "What information do I carry?"

**Step 2:** Compute attention scores and apply them:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The $QK^T$ computes pairwise similarity between all queries and keys. Division by $\sqrt{d_k}$ prevents the dot products from growing too large (which would push softmax into saturation). Softmax normalizes scores to probabilities. The result: each token's output is a weighted sum of all values, weighted by relevance.

### Multi-Head Attention

Run $h$ attention heads in parallel, each with its own $W_Q, W_K, W_V$ projections (of dimension $d_k = d_{model}/h$). Concatenate outputs and project:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

Each head can learn different attention patterns — one head might attend to syntactic relationships, another to semantic similarity, another to positional proximity.

### Positional Encoding

Self-attention is permutation-invariant — it does not know token order. Positional encodings inject position information.

**Sinusoidal (original):** $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$, $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$. Different frequencies encode different scales of position.

**Learned (modern):** Trainable embedding per position. More flexible, used in GPT-2 and later models. RoPE (Rotary Position Embeddings) in LLaMA encodes relative positions through rotation matrices — generalizes better to unseen sequence lengths.

### Encoder vs. Decoder

**Encoder (BERT-style):** Bidirectional self-attention — each token sees all other tokens. Good for understanding tasks (classification, NER, sentence similarity). Not used for generation.

**Decoder (GPT-style):** Causal (masked) self-attention — each token can only see previous tokens. Used for generation. A mask sets attention weights to $-\infty$ for future positions before the softmax, ensuring they get zero weight.

**Encoder-Decoder (T5, original transformer):** Encoder processes input with bidirectional attention. Decoder generates output with causal self-attention plus cross-attention to encoder outputs. Used for seq2seq tasks (translation, summarization).

### Why Transformers Dominate

1. **Parallelizable:** All attention computations happen simultaneously (unlike RNNs which must process sequentially). Training scales with GPU count.
2. **Long-range dependencies:** Any token can attend to any other token in one layer (RNNs need information to travel through many sequential steps).
3. **Scalable:** Performance improves predictably with more parameters and more data (scaling laws).

---

## 3.6 Pre-training & Fine-tuning

### Pre-training Objectives

**Masked Language Modeling (MLM — BERT):** Randomly mask 15% of tokens, train the model to predict the masked tokens from context. Bidirectional — uses both left and right context. Produces representations good for understanding.

**Causal Language Modeling (CLM — GPT):** Predict the next token given all previous tokens. Left-to-right only. The pre-training objective for all modern generative LLMs. Simple, scalable, and remarkably effective — next-token prediction on trillions of tokens produces emergent capabilities.

### Fine-tuning

Take a pre-trained model and continue training on task-specific data with a small learning rate:

1. Load pre-trained weights (captures general language knowledge)
2. Add or modify the output layer if needed (e.g., classification head)
3. Train on labeled task data with learning rate 10-100x smaller than pre-training
4. The model adapts its representations to the specific task while retaining general knowledge

**Why it works (transfer learning):** Pre-training captures linguistic knowledge (syntax, semantics, world knowledge) that transfers across tasks. Fine-tuning on 10K labeled examples leverages representations learned from trillions of tokens.

---

## 3.7 Parameter-Efficient Fine-Tuning (PEFT)

### The Problem

Full fine-tuning updates every parameter in the model. For a 8B parameter model in FP16, this means:
- **Model weights:** 16 GB
- **Optimizer states (Adam):** 32 GB (two momentum buffers, one per parameter)
- **Gradients:** 16 GB
- **Total:** ~64 GB minimum — more than a single consumer GPU

For 70B models, you need hundreds of gigabytes of GPU memory just for fine-tuning. This is impractical for most researchers and companies.

### LoRA (Low-Rank Adaptation)

The key insight: weight updates during fine-tuning are low-rank — they live in a small subspace of the full parameter space. Instead of updating the full weight matrix $W \in \mathbb{R}^{d \times d}$, learn a low-rank decomposition of the update:

$$W' = W + \Delta W = W + A \times B$$

Where:
- $W$ is the original pre-trained weight (frozen, not updated)
- $A \in \mathbb{R}^{d \times r}$ — projects down to low-rank space
- $B \in \mathbb{R}^{r \times d}$ — projects back up
- $r \ll d$ — the rank (typically $r = 8, 16, 32$)

**Parameter savings:** For a layer with $d = 4096$ and $r = 16$:
- Full fine-tuning: $4096 \times 4096 = 16.7M$ parameters per layer
- LoRA: $(4096 \times 16) + (16 \times 4096) = 131K$ parameters per layer
- Savings: 128x fewer trainable parameters

**Memory savings:** Only LoRA parameters need optimizer states and gradients. The base model stays in inference mode (no gradient computation needed for frozen weights).

### Rank Parameter

The rank $r$ controls the capacity of the adaptation:
- **$r = 4$:** Very constrained. Good for simple tasks (style transfer, format adaptation).
- **$r = 16$:** Standard default. Balances capacity and efficiency for most tasks.
- **$r = 64$:** High capacity. Approaches full fine-tuning expressiveness for complex tasks.

Higher rank means more parameters, more memory, slower training — but more capacity to capture complex task-specific behavior.

### QLoRA (Quantized LoRA)

Combine LoRA with quantization for extreme memory efficiency:
1. Quantize the base model to 4-bit (NF4 data type — a quantization scheme optimized for normally-distributed neural network weights)
2. Keep LoRA adapters in FP16/BF16 (full precision)
3. During forward pass: dequantize base weights on-the-fly, apply LoRA adapter

**Result:** Fine-tune a 8B model on a single 24GB consumer GPU. Fine-tune a 70B model on a single 48GB A6000. This democratized LLM fine-tuning — previously requiring multi-GPU clusters, now possible on hardware accessible to individuals and small labs.

### Author's Use: LoRA for Reasoning Distillation

In the MTP thesis, I applied LoRA (rank-16) to LLaMA-3.1-8B to distill A* search traces into linear Chain-of-Thought reasoning. The training procedure:

1. Run A* search to find optimal reasoning paths for training problems
2. Format these paths as linear CoT traces (step-by-step reasoning)
3. Fine-tune LLaMA-3.1-8B with LoRA to reproduce these traces
4. At inference, the fine-tuned model generates search-quality reasoning in a single forward pass (no search required)

This is a form of knowledge distillation — the "teacher" is the A* search algorithm, and the "student" is the LoRA-adapted LLM. The fine-tuned model cannot match A* on the hardest problems (where search is necessary), but it matches or exceeds A* on simpler problems while being orders of magnitude faster.

### Other PEFT Methods

**Adapter layers:** Insert small trainable layers between frozen transformer layers. Similar idea to LoRA but adds parameters rather than reparameterizing existing ones.

**Prefix tuning:** Prepend trainable "virtual tokens" to the input. The model attends to these virtual tokens, which learn task-specific context. No modification to model weights at all.

**Prompt tuning:** Simplified prefix tuning — learn a soft prompt (continuous embedding vectors) prepended to the input. Even fewer parameters than prefix tuning.

---

## 3.8 Learning Rate Scheduling

The learning rate is arguably the most important hyperparameter in deep learning. A fixed learning rate is rarely optimal — scheduling adapts it through training for better convergence.

### Warmup

Start with a very small learning rate and increase linearly for the first $N$ steps (typically 1-10% of total training steps).

**Why:** At initialization, model parameters are random. Large updates in random directions can destabilize training permanently. Warmup lets the model find a reasonable region of parameter space before committing to large steps.

For fine-tuning with LoRA, warmup is shorter (parameters are already in a good region from pre-training), but still beneficial.

### Cosine Decay

After warmup, the learning rate follows a cosine curve from its peak value down to near-zero:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t \cdot \pi}{T}\right)\right)$$

**Why cosine?** Early in training, high learning rate allows exploration of the loss landscape. Late in training, low learning rate allows fine-grained convergence to a minimum. The cosine shape provides a smooth transition rather than abrupt drops.

### Complete Schedule: Warmup + Cosine Decay

```
LR
|    /\
|   /  \
|  /    \____
| /          \___
|/               \___
+------------------------→ Steps
  warmup    cosine decay
```

This is the standard schedule for transformer pre-training and fine-tuning. The MTP thesis uses this schedule for LoRA fine-tuning: 100 warmup steps, cosine decay over the remaining training, peak learning rate of 2e-4.

### Why It Matters in Practice

**Too high throughout:** Gradients are large, parameters oscillate, loss spikes, training diverges. For fine-tuning, catastrophic forgetting — the model "forgets" its pre-trained knowledge.

**Too low throughout:** Training makes imperceptible progress. You waste compute without reaching a good solution.

**Fixed but "just right":** Works for some problems, but the optimal learning rate changes as training progresses. Early on, the loss landscape is rough — you need large steps to escape bad regions. Late on, you are near a good minimum — you need small steps to avoid overshooting.

Scheduling gets the best of both regimes automatically.

---

## 3.9 Summary

### The Architecture Lineage

$$\text{Perceptron} \rightarrow \text{MLP} \rightarrow \text{CNN/RNN} \rightarrow \text{Transformer} \rightarrow \text{LLM}$$

Each architecture solved a limitation of the previous: MLPs could not handle spatial structure (CNNs solved it). RNNs could not parallelize or capture long-range dependencies efficiently (Transformers solved it). Transformers at scale, trained on massive data, produced LLMs with emergent capabilities.

### Key Takeaways for Interviews

| Concept | Why It Matters |
|---------|---------------|
| Attention mechanism | The core innovation enabling all modern LLMs. Know the math cold. |
| Vanishing/exploding gradients | Classic question. Know the problem AND the solutions (ReLU, ResNets, clipping). |
| LoRA | Makes fine-tuning accessible. Know the math (low-rank decomposition) and practical details (rank selection, which layers to adapt). |
| Encoder vs. Decoder | Determines what a model can do. BERT = understanding, GPT = generation. |
| Positional encoding | Transformers have no built-in position sense. This is how order is injected. |
| Learning rate scheduling | The difference between training that works and training that diverges. |

### Connections to the Book

- **Chapter 4 (LLMs):** Transformers are the architecture. Autoregressive generation is causal attention applied sequentially. KV-cache stores the K and V matrices from self-attention.
- **Chapter 6 (RAG):** Embedding models (bi-encoders for retrieval) are transformer encoders. Cross-encoders (re-rankers) are encoder-based classifiers.
- **Chapter 10 (Search):** A* search operates over reasoning traces that are generated by transformer decoders. LoRA fine-tuning distills search knowledge back into the transformer.
- **Chapter 7 (Agents):** Tool-use is implemented by training (or prompting) a decoder transformer to output structured function calls alongside natural language.

Transformers are not just one architecture among many. They are the architecture — the substrate on which modern AI is built. Understanding them deeply, from attention math to training dynamics to efficient adaptation, is the single highest-leverage investment you can make for AI interviews in 2025-2026.
