# Unhidra_gpt.py
# Minimal GPT-style decoder block for our custom BPE tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTConfig:
    def __init__(
        self,
        vocab_size,
        n_ctx=256,
        n_embd=256,
        n_head=8,
        n_layer=8,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout


class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.att = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class UnhidraGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.n_ctx, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.config = config

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.n_ctx:]

            logits = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)

        return idx
