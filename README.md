U-ai

A simple transformer AI engine written in pure Rust. No external models, no machine learning crates, everything built from scratch.

Current features:
- custom .tmod model file
- random tiny transformer
- token embeddings
- positional embeddings
- single-head attention
- feed-forward network
- softmax decoding
- autoregressive generation
- works offline in Termux

How to run:
cargo build --release
./target/release/local_ai

If the model file is missing, it generates a new one.

Next steps:
- layernorm and residuals
- multi-head attention
- causal masking
- tokenizer and training
- better model tools
