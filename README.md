U-ai

A small Transformer engine written in pure Rust.  
Runs fully offline with no external ML crates or models.

Current features:
- custom .tmod model format
- embeddings + positional embeddings
- multi-head attention
- pre-LayerNorm
- residual connections
- feed-forward layer
- softmax decoding
- works offline in Termux or Linux

How to run:
cargo build --release
./target/release/local_ai

If the model file is missing, a new one is generated.

Next steps:
- causal masking
- training loop
- tokenizer
