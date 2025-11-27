import torch
from bpe_tokenizer_fixed import BPETokenizer
from Unhidra_gpt import UnhidraGPT, GPTConfig

MODEL_PATH = "/home/unhidra/U-ai/py_model/unhidra_model.pt"
TOKENIZER_PATH = "/home/unhidra/U-ai/bpe/tokenizer.json"

def load_model():
    data = torch.load(MODEL_PATH, map_location="cpu")

    config = GPTConfig(**data["config"])
    model = UnhidraGPT(config)
    model.load_state_dict(data["model"])
    model.eval()
    return model

def chat(model, tokenizer):
    print("UnhidraGPT ready.")
    history = ""

    while True:
        try:
            user_msg = input("You: ")
        except EOFError:
            break

        history += user_msg + "\nAssistant: "

        ids = tokenizer.encode(history)
        idx = torch.tensor([ids], dtype=torch.long)

        out = model.generate(idx, max_new_tokens=60)[0].tolist()
        reply = tokenizer.decode(out[len(ids):]).strip()

        print("AI:", reply)
        history += reply + "\n"

if __name__ == "__main__":
    tokenizer = BPETokenizer(TOKENIZER_PATH)
    model = load_model()
    chat(model, tokenizer)
