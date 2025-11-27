import os
import yaml
import torch
from corpora_loader import load_all_corpora
from corpora_loader import load_all_corpora
import torch.nn as nn
from corpora_loader import load_all_corpora
from corpora_loader import load_all_corpora
from torch.utils.data import Dataset, DataLoader
from tokenizer.bpe_tokenizer import BPETokenizer
from model.unhidra_gpt import UnhidraGPT, GPTConfig


class TextDataset(Dataset):
    def __init__(self, tokenizer, corpus_dir, block_size):
        all_text = load_all_corpora("/home/unhidra/U-ai/corpora")
        for fname in os.listdir(corpus_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
                    all_text += f.read() + "\n"

        ids = tokenizer.encode(all_text)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def train():
    config_path = os.path.expanduser("~/U-ai/config/config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer = BPETokenizer(cfg["tokenizer"]["path"])
    corpus_dir = cfg["paths"]["corpora"]

    block_size = cfg["training"]["block_size"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    max_steps = cfg["training"]["max_steps"]

    dataset = TextDataset(tokenizer, corpus_dir, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    print(f"Dataset tokens: {len(dataset.data)}")

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_ctx=block_size,
        n_embd=cfg["model"]["n_embd"],
        n_head=cfg["model"]["n_head"],
        n_layer=cfg["model"]["n_layer"],
        dropout=cfg["model"]["dropout"]
    )

    model = UnhidraGPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    step = 0

    for epoch in range(999_999):
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
                print(f"step {step}  loss {loss.item():.4f}")

            if step >= max_steps:
                out = os.path.expanduser("~/U-ai/model/unhidra_model_v2.pt")
                torch.save({
                    "model": model.state_dict(),
                    "config": vars(model_cfg),
                }, out)
                print(f"Training complete. Saved to {out}")
                return


if __name__ == "__main__":
    train()
