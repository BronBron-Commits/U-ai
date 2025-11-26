U-ai

A small Rust-based AI engine with optional external entropy seeding.

Overview
This project implements a minimal neural network and tokenizer in pure Rust.  
It can optionally pull entropy from an external entropy source (such as a lava-lamp webcam appliance) and use it to seed a ChaCha20 RNG for sampling tokens.

Features
• Pure Rust inference code (no Python dependencies)  
• Simple feedforward network architecture  
• Byte-level tokenizer  
• Optional entropy-driven RNG sampler  
• Lightweight and portable for experimentation on mobile or embedded devices

Building
cargo build --release

Running
./target/release/uai

Entropy Integration
If an entropy server is reachable at:
http://127.0.0.1:8080/entropy

The model’s random sampling uses ChaCha20 seeded from real-world entropy.  
If the server is unavailable, it falls back to a local random seed.

Project Structure
src/
  model/        – neural network and layers
  tokenizer/    – byte tokenizer
  entropy/      – fetch and sampler modules
  main.rs       – entry point