# Transformer Architecture: Engineering Foundations

The shift from RNNs to Transformers was a shift from **sequential processing** to **global parallelization**. As an AI engineer, I am focusing on the Transformer as the "operating system" for modern SLMs.

---

## 1. The Core Engine: Scaled Dot-Product Attention
The fundamental breakthrough is the ability to weigh the importance of different tokens in a sequence simultaneously, regardless of distance.

The mechanism is defined by the following operation on Query ($Q$), Key ($K$), and Value ($V$) matrices:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Key Parameters:
* **$QK^T$ (The Score):** Determines the "affinity" between tokens.
* **$\sqrt{d_k}$ (Scaling):** Prevents the dot product from growing too large, which would push the softmax into regions with extremely small gradients.
* **Softmax:** Converts raw scores into a probability distribution, ensuring the model's "focus" sums to $1$.

---

## 2. Multi-Head Attention (MHA)
Instead of a single attention pass, we project $Q, K, \text{ and } V$ into $h$ different subspaces. This allows the model to learn diverse relationships (e.g., one head for syntax, one for semantics).

**Engineering Note:** Standard MHA has $O(n^2)$ complexity. When I build my SLM, I will explore **Grouped-Query Attention (GQA)**, which reduces the KV-cache size and improves inference speed—a critical requirement for production environments.

---

## 3. Position-Wise Feed-Forward Networks (FFN)
Following the attention layer, each token's representation passes through an FFN. This is where the model's "stored knowledge" resides.

$$FFN(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2$$

Most modern architectures (like Llama) swap the standard ReLU activation for **SiLU** or **GeLU** to allow for smoother gradient flow during backpropagation.

---

## 4. Normalization and Stability
To ensure training stability across many layers, we use Layer Normalization. 

* **Pre-Norm:** Most modern LLMs use Pre-Norm (normalizing before the attention/FFN blocks) because it allows for much more stable training of deep networks.
* **RMSNorm:** I will be implementing **Root Mean Square Layer Normalization**, which is computationally cheaper than standard LayerNorm as it ignores the mean centering.

---

## 5. Why this matters for SLMs
Building a Small Language Model (SLM) isn't just about shrinking a big model. It’s about:
1.  **Data Quality:** Smaller models require "cleaner" tokens to reach the same level of reasoning.
2.  **Context Efficiency:** Using **Rotary Positional Embeddings (RoPE)** instead of absolute embeddings to help the model generalize better to longer sequences.
3.  **Compute-Optimal Training:** Aligning with **Chinchilla Scaling Laws** to ensure I'm not over-training or under-training given my parameter count.