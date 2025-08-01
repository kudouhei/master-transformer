import torch
import os

# accelerate 是 Hugging Face 开发的一个库，它简化了 PyTorch 分布式训练的过程
from accelerate import Accelerator

# bitsandbytes 是一个用于优化深度学习模型训练的库，主要提供高效的8-bit优化器（如8-bit Adam）和量化功能，
# 可以显著减少显存占用同时保持模型性能
# import bitsandbytes as bnb

# define GPT architecture
class GPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = torch.nn.Embedding(block_size, n_embed)
        self.blocks = torch.nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embed)
        self.lm_head = torch.nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Ensure positional indices are on the same device as idx
        pos_emb = self.positional_embedding(torch.arange(T, device=idx.device))
        tok_emb = self.token_embedding(idx)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx

# define the building blocks of the model
class Block(torch.nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.ln1 = torch.nn.LayerNorm(n_embed)
        self.ln2 = torch.nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.projection = torch.nn.Linear(head_size * num_heads, n_embed)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(n_embed, head_size, bias=False)
        self.query = torch.nn.Linear(n_embed, head_size, bias=False)
        self.value = torch.nn.Linear(n_embed, head_size, bias=False)

        # Ensure tril buffer is always on the correct device
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)), persistent=False)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Move tril to the same device as x if needed
        tril = self.tril[:T, :T]
        if tril.device != x.device:
            tril = tril.to(x.device)

        attention = q @ k.transpose(-2, -1) * k.shape[-1] ** 0.5
        attention = attention.masked_fill(
            tril == 0, float("-inf")
        )

        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        v = self.value(x)
        out = attention @ v
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Helper functions for training
def encode(string):
    return [uutt2int[c] for c in string]

def decode(line):
    return "".join([int2utt[i] for i in line])

def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# train the model
if __name__ == "__main__":
    batch_size = 64
    block_size = 256
    max_iters = 500
    eval_interval = 100 # 每500个迭代进行一次评估
    learning_rate = 3e-4 # 学习率

    eval_iters = 200 # 评估200次
    n_embed = 384 # 嵌入维度
    n_head = 6 # 多头注意力机制的头部数量
    n_layer = 6 # Transformer块的数量
    dropout = 0.2 # 丢弃率

    accelerator = Accelerator()
    # Use MPS if available, else CUDA, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    doing_quantization = False  # Change to True if imported bitsandbytes

    # dataset
    with open("./data/crimeandpunishment.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Character-based pseudo-tokenization
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    uutt2int = {ch: i for i, ch in enumerate(chars)}
    int2utt = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # model
    model = GPT()
    model = model.to(device)

    print("Instantiated Model")
    print(
        sum(param.numel() for param in model.parameters()) / 1e6,
        "Model parameters",
    )

    # Optimizer
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=learning_rate)
        if not doing_quantization
        else bnb.optim.Adam(model.parameters(), lr=learning_rate)
    )

    print("Instantiated Optimizer")

    model, optimizer, train_data = accelerator.prepare(
        model, optimizer, train_data
    )
    print("Prepared model, optimizer, and data")

    # Training block
    for iter in range(max_iters):
        print(f"Running Epoch {iter}")

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"| step {iter}: train loss {losses['train']:.4f} "
                f"| validation loss {losses['val']:.4f} |"
            )

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    # Create model directory
    model_dir = "./models/scratchGPT/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model
    model_path = model_dir + "model.pt"
    torch.save(
        model.state_dict(),
        model_path,
    )

    # Load the saved model
    loaded_model = GPT().to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))

    # Test the loaded model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(
        decode(
            loaded_model.generate(context, max_new_tokens=500)[0].tolist()
        )
    )
