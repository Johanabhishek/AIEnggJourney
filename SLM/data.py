import os
import torch
import tiktoken

def get_data(filepath="MovieCorpus.txt", split_ratio=0.9):
    """
    Reads the text file, encodes it using GPT-2 tiktoken,
    and returns a train and validation PyTorch tensor.
    """
    print(f"Loading data from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print("Encoding data...")
    enc = tiktoken.get_encoding("gpt2")
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train data size: {len(train_data)} tokens")
    print(f"Validation data size: {len(val_data)} tokens")
    
    return train_data, val_data

def get_batch(data, batch_size=4, block_size=8, device='cpu'):
    """
    Generates a small batch of data of inputs x and targets y.
    """
    # random starting indices for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

if __name__ == "__main__":
    # Quick test when running the script directly
    train_data, val_data = get_data()
    
    x, y = get_batch(train_data, batch_size=4, block_size=8)
    print("Sample Batch Input (x):")
    print(x)
    print("Sample Batch Target (y):")
    print(y)
