# Plan: Build an SLM from Scratch (Learning-First Approach)

## Context

The user has a movie dialogue corpus (MovieCorpus.txt, 16MB, ~4.67M tokens) tokenized with GPT-2 via tiktoken. The goal is to build a Small Language Model from scratch to **learn**, using an LLM as a tutor (not a coder). Documentation goes into a git repo.

---

## Current State

- `tokenizer.py` — tokenizes corpus, prints stats, **does not save tokens to disk**
- `slm.ipynb` — empty placeholder
- `MovieCorpus.txt` — raw data
- Stack: Python 3.13, tiktoken, numpy, tqdm, jupyter (no torch yet)

---

## Implementation Plan

### Step 0: Environment & Git Setup
**Goal:** Establish a reproducible foundation.

1. Install PyTorch (CPU or CUDA if GPU available): `pip install torch`
2. `pip install matplotlib` (for loss curves)
3. `git init` the project, add `.gitignore` (exclude `venv/`, `*.npy`, large data)
4. Write a `README.md` — just the problem statement and goals for now
5. **Commit:** `feat: initial project setup`

**LLM Use:** Ask Claude/ChatGPT: *"Why do language models need PyTorch/autograd instead of just numpy?"*

---

### Step 1: Save Tokens to Disk
**Goal:** Produce a binary token file you can load fast during training.

**File:** `prepare_data.py`

```python
# Load corpus → tokenize → split 90/10 train/val → save as .npy
train_tokens = np.array(tokens[:split])    # shape: (N,)  dtype: uint16
val_tokens   = np.array(tokens[split:])
np.save("train.npy", train_tokens)
np.save("val.npy", val_tokens)
```

- Use `dtype=np.uint16` (GPT-2 vocab is 50257, fits in uint16, saves space)
- **Commit:** `feat: save tokenized data to disk (train/val split)`

**LLM Use:** Ask *"Why do we save tokens as uint16? What's the tradeoff with uint32?"*

---

### Step 2: Dataset & DataLoader
**Goal:** Understand how training data is fed into a model.

**File:** `dataset.py`

Implement a `TextDataset` class that:
- Memory-maps the `.npy` file (don't load all 4.67M tokens into RAM at once)
- Returns `(x, y)` pairs where `y = x shifted by 1` (next-token prediction)
- Uses a `block_size` (context length, e.g. 256)

```python
class TextDataset:
    def __init__(self, path, block_size):
        self.data = np.load(path, mmap_mode='r')
        self.block_size = block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y
```

- **Commit:** `feat: dataset class with block_size windowing`

**LLM Use:** Ask *"Why is the target y just x shifted by one token? How does this teach a model to predict next tokens?"*

---

### Step 3: Bigram Model (Baseline)
**Goal:** Build the simplest possible language model — one that only looks at the previous token. This is your sanity check before building anything complex.

**File:** `model_bigram.py`

```python
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.embedding(x)           # (B, T, vocab_size)
        loss = F.cross_entropy(logits, targets) if targets else None
        return logits, loss
```

Train it for 1000 steps. Generated text will be garbage — **that's the point**. This gives you the baseline loss to beat.

- **Commit:** `feat: bigram baseline model + training loop`

**LLM Use:** Ask *"What is cross-entropy loss in the context of language modeling? What does a loss of 10.8 (random guessing for vocab_size=50257) vs 2.5 mean?"*

---

### Step 4: The Transformer — Build Block by Block
**Goal:** Understand every component of a GPT-style transformer by building each piece manually.

**File:** `model.py` (build incrementally, one class at a time)

#### 4a. Token + Positional Embeddings
```python
self.tok_emb = nn.Embedding(vocab_size, n_embd)
self.pos_emb = nn.Embedding(block_size, n_embd)
```
**LLM Use:** Ask *"Why does a transformer need positional embeddings? Doesn't attention already see all tokens?"*

#### 4b. Self-Attention Head (single head first)
```python
class Head(nn.Module):
    # Q, K, V projections
    # scores = Q @ K.T / sqrt(d_k)
    # apply causal mask (future tokens = -inf)
    # weights = softmax(scores)
    # out = weights @ V
```
**LLM Use:** Ask *"Draw me the math of one attention head step by step. Why divide by sqrt(d_k)?"*
**Commit after each sub-component.**

#### 4c. Multi-Head Attention
Concatenate `n_heads` independent attention heads.
**LLM Use:** Ask *"Why use multiple heads instead of one big head?"*

#### 4d. Feed-Forward Network
```python
class FFN(nn.Module):
    # Linear → ReLU/GELU → Linear
    # hidden_dim = 4 * n_embd  (standard GPT ratio)
```

#### 4e. Transformer Block (Attention + FFN + LayerNorm + Residuals)
```python
class Block(nn.Module):
    x = x + self.attn(self.ln1(x))   # residual around attention
    x = x + self.ffn(self.ln2(x))    # residual around FFN
```
**LLM Use:** Ask *"Why apply LayerNorm BEFORE attention (pre-norm) instead of after?"*

#### 4f. Full GPT Model
Stack N blocks + final linear projection to vocab_size.

**Recommended starter config:**
```python
n_embd     = 128
n_heads    = 4
n_layers   = 4
block_size = 256
dropout    = 0.1
# ~3M parameters
```

- **Commit:** `feat: full transformer architecture`

---

### Step 5: Training Loop
**Goal:** Train the model and watch loss decrease.

**File:** `train.py`

Key elements:
- **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=0.1`)
- **Batch size:** 32–64 (adjust to fit memory)
- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- **Eval loop:** every 500 steps, compute val loss, print both
- **Loss logging:** append to a list, plot with matplotlib at the end
- **Checkpoint saving:** `torch.save(model.state_dict(), 'checkpoint.pt')` at best val loss

```python
for step in range(max_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

- **Commit:** `feat: training loop with eval and checkpointing`

**LLM Use:** Ask *"What is gradient clipping and why do transformers need it more than simple networks?"*

---

### Step 6: Text Generation (Inference)
**Goal:** Sample from the trained model.

**File:** `generate.py`

```python
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]         # crop context
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # last token, scale
        # optional: zero out all logits except top-k
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
    return idx
```

- **Commit:** `feat: text generation with temperature + top-k sampling`

**LLM Use:** Ask *"What does temperature do to the probability distribution? When would you use temperature=0.5 vs 1.5?"*

---

### Step 7: Experiment & Iterate
Try changing one thing at a time and observe:
- Double `n_layers` → does val loss improve?
- Increase `block_size` → does the model produce more coherent long sentences?
- Try `n_embd=256` → how much slower does training become?
- Add a learning rate scheduler (cosine decay)

**Commit each experiment** with a note on what changed and what happened to the loss.

---

## Documentation Strategy (Git-Based)

**Commit discipline:**
- One commit per step above — each commit = one learnable concept
- Commit message format: `feat: <what>` or `learn: <concept explored>`
- Add a comment block at the top of each file: *what this file does and what concept it teaches*

**Jupyter Notebook (`slm.ipynb`):**
- Use it as your **learning journal**, not just code
- Each cell group: markdown explaining the concept → code → output
- Structure: Data → Model → Train → Generate → Experiments
- Think of it as the narrative version of your `.py` files

**README.md evolution:**
- Step 0: Problem statement
- After Step 3: Baseline results (bigram loss)
- After Step 5: Training curves image
- After Step 6: Sample generated text
- Final: Architecture diagram, hyperparameters, what you learned

**Repo structure to aim for:**
```
slm-build/
├── README.md
├── slm.ipynb          ← learning journal
├── prepare_data.py
├── dataset.py
├── model_bigram.py
├── model.py
├── train.py
├── generate.py
├── MovieCorpus.txt
├── train.npy          ← gitignored (large)
├── val.npy            ← gitignored
└── checkpoint.pt      ← gitignored
```

---

## Where to Use an LLM (Learning Guide)

| When | What to ask |
|------|------------|
| Before coding attention | "Explain self-attention with a concrete 3-token example, step by step with numbers" |
| When loss doesn't decrease | "My transformer loss is stuck at X after 1000 steps. Here's my training code. What could cause this?" |
| After training | "My train loss is 2.1 but val loss is 4.5. What is happening and how do I fix it?" |
| When confused by a paper | Paste the equation and ask "explain this line to me like I'm building it in PyTorch" |
| **Never** | Ask it to write entire files for you. Write the code yourself, use LLM to understand it |

---

## Verification / Testing Each Step

- **Step 1:** `python prepare_data.py` → `train.npy` and `val.npy` appear, sizes printed
- **Step 2:** `python dataset.py` → print a single `(x, y)` batch and verify y = x shifted
- **Step 3:** Bigram loss should drop from ~10.8 to ~5-6 in 1000 steps
- **Step 4:** Count model parameters — confirm ~3M for starter config
- **Step 5:** Val loss should reach ~3.5–4.5 with the starter config on this corpus
- **Step 6:** Generated text should look like broken movie dialogue (coherent short phrases)
