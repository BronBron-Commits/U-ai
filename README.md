# U-ai

U-ai is a small experimental AI project written in Rust.  
It provides a lightweight text interface and can use an external entropy source when available.

## Features

- Custom tokenizer using SentencePiece.
- Simple terminal chat interface.
- Optional entropy integration through a named pipe.
- Compact and easy-to-read project structure.

## Running

To run with an initial message:

    cargo run -- "your message"

To start interactive mode:

    cargo run

## Entropy Integration

If a named pipe exists at:

    /tmp/unhidra_entropy.pipe

U-ai will read one byte from it before producing a response.  
If no pipe exists, U-ai falls back to pseudorandom behavior.

## Project Structure

- src/tokenizer – SentencePiece-related code.
- src/entropy.rs – optional entropy helper.
- src/main.rs – CLI entry point.
- src/lib.rs – main engine.

## Requirements

- Rust toolchain  
- SentencePiece C libraries installed  
- Optional external entropy source writing to `/tmp/unhidra_entropy.pipe`

## License
