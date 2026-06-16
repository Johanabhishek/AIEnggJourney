# Small Language Model (SLM) Build

A PyTorch-based, from-scratch implementation of a Small Language Model (Decoder-only Transformer architecture).

## Overview
This project demonstrates the complete pipeline for building and training a custom language model based on the GPT architecture. It includes data preparation, tokenization, the neural network architecture, and the training loop.

## Features
- **Data Pipeline (`data.py`)**: Uses OpenAI's `tiktoken` (GPT-2 encoding) to tokenize custom text datasets and dynamically generate training/validation batches.
- **Model Architecture (`model.py`)**: A fully implemented Transformer model including:
  - Multi-Head Self Attention
  - Feed-Forward Networks
  - Positional Encodings
  - Layer Normalization
- **Training Loop (`train.py`)**: Implements an `AdamW` optimization loop with periodic validation loss estimation and a text generation demonstration.

## Setup
1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install PyTorch and other dependencies:
   ```bash
   pip install torch tiktoken
   ```
4. Place your training data in a `MovieCorpus.txt` file (or update the filename in `data.py`).

## Usage
Run the training script to begin training the model:
```bash
python train.py
```
The model will output its training loss trajectory and generate a sample text sequence upon completion.
