# U-ai

A small modular AI experiment written in Rust with a clean architecture:

- SentencePiece tokenization via our custom FFI bridge  
- Modular dataset system  
- Pluggable forward/inference core  
- Expandable training engine  
- Dataset weighting and multi-source learning planned  

## Current Status
- SentencePiece C++ ↔ Rust bridge is working
- Rust tokenizer layer built on top of the FFI
- Model initialization, inference skeletons, and training stubs running

## Goals
- Proper weighted multi-source dataset ingestion
- A small usable transformer-style model
- Configurable training loops
- Compact inference engine

## Structure
- `cpp/` – SentencePiece C++ bridge
- `src/tokenizer/` – Rust tokenizer using the bridge
- `src/model/` – WIP model and training logic
- `src/infer/` – Forward pass + inference utilities
- `src/dataset/` – Dataset registry, sampling, weighting (next major step)

## Requirements
- Rust stable
- libsentecepiece (C++)
- C++17 compiler


