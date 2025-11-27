from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
import json

corpus_dir = os.path.expanduser("~/U-ai/corpora")

files = []
for fname in os.listdir(corpus_dir):
    if fname.endswith(".txt"):
        files.append(os.path.join(corpus_dir, fname))

# Create byte-level BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=50000,
    special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
)

tokenizer.train(files, trainer)

out_dir = os.path.expanduser("~/U-ai/bpe")
os.makedirs(out_dir, exist_ok=True)

# Save as tokenizer.json (the correct file format)
tokenizer.save(os.path.join(out_dir, "tokenizer.json"))

print("BPE tokenizer.json generated!")
