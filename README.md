U-ai
====

U-ai is a fully custom, from-scratch language model project. It contains the following major components:

1. BPE tokenizer built using the HuggingFace tokenizers library
2. PyTorch GPT architecture (small model for now, but expandable)
3. Training pipeline written in Python
4. Custom corpus loader and dataset builder
5. Inference script for running the model interactively
6. Experimental Rust prototypes from the early phase of the project

Project Structure
-----------------
bpe/
    tokenizer.json      (BPE model generated from corpora)
    vocab.json          (intermediate files from the tokenizer tool)
    merges.txt

corpora/
    Base text files used for training

py_model/
    Unhidra_gpt.py      (GPT model architecture)
    train_unhidra.py    (training loop)
    run_unhidra.py      (interactive inference)
    bpe_tokenizer_fixed.py
    unhidra_model.pt    (trained model weights)

Rust prototypes are in the root of the repo. They represent early tokenizers and N-gram engines.

Requirements
------------
Python 3.10+
PyTorch
HuggingFace tokenizers
NumPy

A working installation example:

python3 -m venv uai-env
source uai-env/bin/activate
pip install torch tokenizers numpy

Training
--------
Run the training script to build the model:

python3 py_model/train_unhidra.py

Training output includes loss values every 50 steps.
Model weights are saved to:

U-ai/py_model/unhidra_model.pt

Running the Model
-----------------
After training completes, start the interactive chat:

python3 py_model/run_unhidra.py

This loads:
- the trained weights
- the GPT architecture
- the BPE tokenizer

Purpose
-------
The goal is to build a personal GPT-like system entirely from scratch. This project is meant to be simple, understandable, and fully modifiable. Future improvements include:

- personality tuning
- larger datasets
- deeper model architectures
- front-end UI
- embedding support
- GPU acceleration when available

Notes
-----
This repository contains large files (model weights). GitHub may warn about file sizes. SSH is used for authentication.

Updates will continue as the system evolves.
