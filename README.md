U-ai

A compact Rust neural network experiment with external entropy integration.

Overview

U-ai is a minimal Rust-based neural network used for experimentation and growth. It now supports sampling tokens using an external entropy source (like the lava-lamp webcam entropy appliance) to improve non-deterministic behavior.

Entropy Integration

The model can read entropy chunks from an HTTP endpoint and uses them to seed a ChaCha20 RNG. If entropy is unavailable, it falls back to local randomness.

Usage

cargo build --release
./target/release/uai

Entropy Endpoint

U-ai expects an entropy URL at:
http://127.0.0.1:8080/entropy

If running on a different device, adjust the URL in:
src/entropy_sampler.rs

Project Structure

src/
  model/                — neural network code
  entropy_sampler.rs    — RNG seeded by external entropy
  entropy/client.rs     — fetches entropy via HTTP
  tokenizer/            — byte-level tokenizer
  layers/               — tensor and linear layers

Goal

Keep the system small, understandable, and hackable while experimenting with improved randomness, sampling strategies, and offline AI behavior.
