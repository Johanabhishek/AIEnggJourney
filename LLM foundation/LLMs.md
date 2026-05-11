# LLMs — The Basics

If you're getting into AI engineering, understanding Large Language Models is non-negotiable. This covers what they are, how they work, and what concepts actually matter.

---

## What is an LLM?

A Large Language Model is an AI model trained on massive amounts of text to understand and generate language. It learns patterns from billions of documents — books, websites, code, papers — and uses that to predict and generate text.

Examples: GPT-4, Claude, Gemini, LLaMA.

---

## How They Work

The core task is simple: **predict the next token**.

Given "The cat sat on the...", the model predicts the next most likely word. Do this billions of times across trillions of words during training, and the model starts to understand grammar, facts, reasoning, and context.

### Tokens
LLMs don't read word by word. Text is split into **tokens** — chunks of ~3-4 characters. "unbelievable" becomes ["un", "believ", "able"]. This matters because models have a **context window** — a hard limit on how many tokens they can process at once.

---

## The Transformer Architecture

All major LLMs are built on the **Transformer** architecture (Google, 2017 — "Attention Is All You Need").

The key idea: **self-attention**. Instead of reading text left to right, the model looks at all words simultaneously and figures out which ones relate to each other.

Example: *"The trophy didn't fit because it was too big."* — Self-attention is what tells the model "it" refers to the trophy, not the suitcase.

The flow looks like this:

```
Input text
  -> Tokenization
  -> Embeddings (tokens to vectors)
  -> Transformer blocks (attention + feed-forward layers)
  -> Output (next token probabilities)
```

---

## Pre-training vs Fine-tuning

**Pre-training** is the expensive phase. The model is trained on a huge general corpus. It learns language, knowledge, and reasoning. Costs millions of dollars. Only big labs do this.

**Fine-tuning** is what you'll actually do as an engineer. You take a pre-trained model and train it further on specific data or instructions:

- **Instruction tuning** — teaches the model to follow instructions
- **RLHF** — human raters score outputs, model learns to produce preferred responses (how ChatGPT and Claude are trained)
- **Domain fine-tuning** — train on medical, legal, or any specialized data

---

## Prompting

How you write your input directly affects output quality.

**Zero-shot** — just give the task, no examples:
```
Translate to French: "I love machine learning."
```

**Few-shot** — give a few examples first to show the pattern:
```
Review: "Amazing food!" -> Positive
Review: "Worst experience." -> Negative
Review: "It was fine." -> ?
```

**Chain-of-thought** — ask it to reason step by step:
```
If a train travels 60 mph for 2.5 hours, how far? Think step by step.
```
This significantly improves accuracy on reasoning tasks.

---

## Key Concepts

**Temperature** — controls randomness. 0 = deterministic, always picks the most likely token. Higher = more creative, can get incoherent.

**Context window** — how many tokens the model can see at once. Think of it as working memory. GPT-4o has 128K tokens, Claude 3.5 has 200K.

**Embeddings** — text converted to numerical vectors. Similar meanings end up close together in vector space. Powers semantic search and RAG.

**Hallucinations** — the model generates confident but factually wrong output. It's a next-token predictor, not a fact database. A major production challenge.

**Parameters** — the weights learned during training. GPT-3 had 175B, GPT-4 is estimated at 1T+. More parameters generally means more capability.

---

## Popular Models

| Model | By | Open Source | Context | Best For |
|---|---|---|---|---|
| GPT-4o | OpenAI | No | 128K | General use, coding |
| Claude 3.5 | Anthropic | No | 200K | Reasoning, long docs |
| Gemini 1.5 Pro | Google | No | 1M | Long context tasks |
| LLaMA 3 | Meta | Yes | 128K | Self-hosting, research |

---

## Real-World Use Cases

- **Code generation** — Copilot, Cursor, Claude Code
- **RAG systems** — connect LLMs to private/internal data via retrieval
- **Summarization** — condense documents, reports, papers
- **Agents** — LLMs that use tools and take multi-step actions autonomously
- **Semantic search** — search by meaning, not just keywords
- **Data analysis** — natural language to SQL, chart interpretation

---

## Limitations

**Hallucinations** — always validate outputs in critical applications.

**Bias** — models learn from human data and inherit human biases.

**Knowledge cutoff** — models don't know about events after their training date unless given tools like web search.

**Cost** — API calls at scale add up fast. Smaller, faster models are often better for production.

**Privacy** — don't send sensitive data to external APIs without checking the provider's data policies.

---

## What to Learn Next

1. OpenAI / Anthropic API — start building
2. Prompt engineering — go deeper
3. LangChain or LlamaIndex — LLM app frameworks
4. RAG — connect models to your own data
5. Vector databases — Pinecone, Chroma, pgvector
6. Fine-tuning — customize models on your data
7. AI agents — tool use, multi-step reasoning
8. Evaluation — how to measure model performance in production
