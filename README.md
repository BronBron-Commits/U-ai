U-ai is an experimental small language model written in Rust and designed to run locally in Termux or Debian. The goal of the project is to build a simple conversational system that trains on small text corpora without external dependencies.

The model uses a whitespace tokenizer and a 2-gram probability table. Training merges three sources of text:
1. dataset.txt
2. modules/github/*.txt
3. corpora/*.txt

The merged text is tokenized and converted into a vocabulary. A 2-gram table is built by counting token pairs and normalizing each row into probability distributions. The final model is saved as toy_model.tmod.

The chat mode loads the model and generates responses by predicting the next token based on the previous one. This produces simple local text generation suitable for small experiments.

Project structure:
src/
  main.rs
  chat.rs
  tokenizer/
  train.rs
  llm_engine.rs
modules/github/
corpora/
dataset.txt

Usage:
cargo build --release
./target/release/uai --train
./target/release/uai --chat

The project is intentionally minimal and modular so new corpora, modules, or training logic can be added without major refactoring.
