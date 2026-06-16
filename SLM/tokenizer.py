import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm

with open("MovieCorpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:1000])
print(f"Total characters: {len(text)}")

enc = tiktoken.get_encoding("gpt2")

tokens = enc.encode(text)
print(f"total tokens: {len(tokens)}")
print(f"first 20 tokens: {tokens[:30]}")

print(enc.decode(tokens[:20]))
print(enc.decode([5216]))
print(enc.decode([314]))