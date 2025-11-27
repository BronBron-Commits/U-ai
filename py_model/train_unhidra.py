import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from Unhidra_gpt import UnhidraGPT, GPTConfig
from bpe_tokenizer_fixed import BPETokenizer


class TextDataset(Dataset):
    def __init__(self, tokenizer, corpus_dir, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size

        all_text = ""
        for fname in os.listdir(corpus_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
                    all_text += f.read() + "\n"

        ids = tokenizer.encode(all_text)
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + 1 + self.block_size]
        return x, y


def train():
    # NEW: only one file required
    tokenizer_json = os.path.expanduser("~/U-ai/bpe/tokenizer.json")

    tokenizer = BPETokenizer(tokenizer_json)
    corpus_dir = os.path.expanduser("~/U-ai/corpora")

    block_size = 256
    batch_size = 16
    lr = 3e-4
    max_steps = 2000

    dataset = TextDataset(tokenizer, corpus_dir, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_ctx=block_size,
        n_embd=256,
        n_head=8,
        n_layer=8,
        dropout=0.1,
    )

    model = UnhidraGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training on device: {device}")
    print(f"Dataset tokens: {len(dataset.data)}")
    print(f"Total steps: {max_steps}")

    step = 0
    for epoch in range(99999):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            loss.backward()
            optimizer.step()

            step += 1
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.4f}")

            if step >= max_steps:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": vars(config),
                    },
                    os.path.expanduser("~/U-ai/py_model/unhidra_model.pt"),
                )
                print("Training complete. Model saved.")
                return


if __name__ == "__main__":
    train()
