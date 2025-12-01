import torch
import os

def export_to_tmod(model_path="../model/unhidra_model_v3.pt", out_path="../u-ai-trained.tmod"):
    model = torch.load(model_path, map_location="cpu")
    weights = []
    biases = []

    # Gather all Linear layer weights and biases
    for name, param in model.items():
        if "weight" in name:
            weights.extend(param.flatten().tolist())
        elif "bias" in name:
            biases.extend(param.flatten().tolist())

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("UAI_TMOD_V1\n")
        f.write(f"weights {len(weights)}\n")
        f.write(" ".join(map(str, weights)) + "\n")
        f.write(f"biases {len(biases)}\n")
        f.write(" ".join(map(str, biases)) + "\n")

    print(f"✅ Exported model to {out_path} ({len(weights)} weights, {len(biases)} biases)")

if __name__ == "__main__":
    export_to_tmod()
