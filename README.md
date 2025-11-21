U-ai

U-ai is a fully modular Transformer inference engine written in Rust. It performs a single-token forward pass using a simplified architecture with support for multi-layer attention and feedforward logic. All computations are handled locally with no dependencies on external runtimes.

Directory structure:

src/
- main.rs               - Entry point
- model/                - Model definition and parameters
- io/                   - Save and load weights to file
- layer/                - Neural network layers
  - attention/          - Q, K, V projections and attention logic
  - ff.rs               - Feedforward network
  - residual.rs         - Residual connection logic

How to run:

1. Build the project:
   cargo build --release

2. Run the binary:
   ./target/release/local_ai

This will output a stream of sampled token IDs based on a randomly initialized model. Training and decoding logic are not yet implemented.