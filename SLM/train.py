import torch
import tiktoken
from data import get_data, get_batch
from model import TransformerLanguageModel, device

# Hyperparameters for Training
batch_size = 16
max_iters = 500
eval_interval = 100
learning_rate = 1e-3
eval_iters = 20

print("Loading dataset...")
train_data, val_data = get_data()

print("Initializing model...")
model = TransformerLanguageModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size=batch_size, device=device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("Starting training loop...")
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch(train_data, batch_size=batch_size, device=device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training finished!")
print("-" * 50)
print("Generating sample text...")

# Generate from the model
enc = tiktoken.get_encoding("gpt2")
# Start with a single <|endoftext|> token (usually id 50256) or just a newline
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
print(enc.decode(generated_tokens))
print("-" * 50)
