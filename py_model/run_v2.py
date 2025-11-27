def clean_text(text):
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    return text.strip()
import os
import torch
from tokenizer.bpe_tokenizer import BPETokenizer
from model.unhidra_gpt import UnhidraGPT, GPTConfig
def main():
    model_path = os.path.expanduser("~/U-ai/model/unhidra_model_v2.pt")
    tok_path = os.path.expanduser("~/U-ai/bpe/tokenizer.json")
    data = torch.load(model_path, map_location="cpu")
    cfg_dict = data["config"]

    cfg = GPTConfig(**cfg_dict)
    tokenizer = BPETokenizer(tok_path)

    model = UnhidraGPT(cfg)
    model.load_state_dict(data["model"])
    model.eval()

    print("UnhidraGPT v2 ready.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    while True:
        prompt = input("You: ")
        if not prompt:
            continue

        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long).to(device)

        out = model.generate(x, max_new_tokens=80)
        decoded = clean_text(tokenizer.decode(out[0].tolist()))

        print("AI:", decoded)


if __name__ == "__main__":
    main()
