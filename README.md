U-ai: Modular GPT Training Pipeline

U-ai is a small language model project designed for experimentation with custom training corpora, modular dataset loading, and simple inference. The project includes a tokenizer, a minimal GPT-style architecture, and a flexible training setup that allows new behavior modules to be added without disturbing existing data.

This repository contains the full training and inference pipeline, but does not include private datasets or model checkpoints.

Features

- Modular corpus system with core data and optional behavior modules.
- Clean BPE tokenizer implementation.
- GPT-style model defined in a compact Python module.
- Training script with loss tracking and checkpoint saving.
- Inference script with temperature, top-k, and top-p sampling.
- Separate loader for assembling multiple corpora automatically.

Repository Structure

U-ai/
  corpora/
    core/
    modules/
  model/
    unhidra_gpt.py
  tokenizer/
    bpe_tokenizer.py
  py_model/
    train_v2.py
    run_v2.py
    corpora_loader.py

Corpus System

Training data is split into two areas:

corpora/core/      Stable baseline conversational data
corpora/modules/   Optional behavior files (casual tone, tech tone, etc.)

The training script automatically loads everything in both folders, allowing new behavior to be added by creating a new module file.

Training

cd py_model
python train_v2.py

The script loads all corpora, trains the model, and writes a checkpoint in the model directory.

Running the Model

cd py_model
python run_v2.py

The model loads the latest checkpoint and runs an interactive chat loop.

Tokenizer

The tokenizer uses a BPE vocabulary stored in:

bpe/tokenizer.json

It supports both encode and decode operations for training and inference.

Notes

- Model weights are not included in the repository.
- Private or personal data should be placed in corpora locally and excluded from Git.
- The repository uses .gitignore to prevent checkpoints or corpora from being pushed.