import torch, torch.nn as nn, torch.optim as optim, os

class TinyUnhidra(nn.Module):
    def __init__(self, vocab_size, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.fc(self.embed(x)).log_softmax(dim=-1)

def main():
    data = open("../data/train.txt", "r", encoding="utf-8").read()
    vocab = sorted(list(set(data)))
    tok2id = {ch: i for i, ch in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"🧠 vocab size: {vocab_size}")

    model = TinyUnhidra(vocab_size)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    tokens = [tok2id[ch] for ch in data]
    x = torch.tensor(tokens[:-1], dtype=torch.long)
    y = torch.tensor(tokens[1:], dtype=torch.long)
    print(f"Training on {len(x)} tokens...")

    for epoch in range(3):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch+1}: loss = {loss.item():.4f}")

    os.makedirs("../model", exist_ok=True)
    torch.save(model.state_dict(), "../model/unhidra_model_v3.pt")
    print("✅ Saved model/unhidra_model_v3.pt")

if __name__ == "__main__":
    main()
