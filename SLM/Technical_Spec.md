# SLM Technical Specification and Implementation Log

Building a Small Language Model (SLM) requires a shift from 'brute force' scaling to 'high-density' parameter optimization. This document logs the architectural choices and data strategies employed for a model in the 100M - 350M parameter range.

---

## 1. Architectural Configuration
To maximize performance within a restricted parameter budget, the following components were prioritized over standard vanilla Transformer blocks:

### Rotary Positional Embeddings (RoPE)
Instead of absolute positional encodings, RoPE is implemented to encode relative positions via rotation matrices. 
- **Benefit:** Better extrapolation to sequence lengths beyond the training window.
- **Learning:** Absolute embeddings fail when the context window shifts; RoPE maintains the 'dot product' similarity across shifts.

### Grouped-Query Attention (GQA)
Standard Multi-Head Attention (MHA) is memory-bound during inference due to the KV-cache.
- **Config:** Using a ratio of 4:1 (4 Query heads for 1 Key/Value head).
- **Learning:** This reduces the memory bandwidth requirement significantly while retaining nearly the same accuracy as MHA.

[Image of Grouped Query Attention architecture]

### RMSNorm (Root Mean Square Layer Normalization)
Replaced standard LayerNorm.
- **Implementation:** Normalization is applied at the start of each block (Pre-Norm).
- **Learning:** RMSNorm is computationally cheaper because it avoids mean-centering. Pre-norm is essential for training stability in deep stacks.

---

## 2. Data Curation Strategy
Small models are highly sensitive to noise. The 'Garbage In, Garbage Out' rule is magnified at the 100M scale.

### Signal-to-Noise Ratio (SNR)
- **Curation:** Focused on high-density reasoning datasets (e.g., Python code, textbook-style explanations).
- **Deduplication:** Used MinHash LSH to remove near-duplicate documents. 
- **Learning:** Redundancy in small datasets leads to overfitting on specific phrasing rather than general logic.

### Tokenization (BPE)
- **Vocab Size:** 32k - 50k tokens.
- **Learning:** A larger vocab reduces sequence length but increases the embedding layer's parameter footprint. For an SLM, a smaller, highly-trained vocabulary is more efficient.

---

## 3. Training and Optimization
### Weight Tying
- **Strategy:** Tied the weights of the input embedding layer and the output linear head.
- **Result:** Reduced total parameter count by ~15-20% without performance degradation.

### SwiGLU Activation Function
- **Formula:** $f(x, W, V, b, c) = 	ext{Swish}_1(xW + b) \otimes (xV + c)$
- **Learning:** SwiGLU outperforms ReLU by providing more expressive non-linearity, though it requires more compute per forward pass.

---

## 4. Hardware and Inference Constraints
### Compute-Optimal Training (Chinchilla Laws)
- **Target:** 20 tokens per parameter. For a 100M model, the target was 2B tokens.
- **Bottleneck:** I/O throughput was the primary bottleneck during local training; utilized pinned memory and multi-worker data loaders to saturate the GPU.

### Precision
- **Strategy:** Trained using BFloat16 to reduce memory footprint and increase throughput on modern hardware while maintaining numerical stability compared to FP16.
