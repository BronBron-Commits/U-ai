use std::io::{stdin, stdout, Write};
use crate::tokenizer::Tokenizer;
use crate::llm_engine::Engine;

pub struct ChatSession {
    pub history: String,
}

impl ChatSession {
    pub fn new() -> Self {
        ChatSession { history: String::new() }
    }

    pub fn run(&mut self) {
        let tok_bytes = std::fs::read("tokenizer.tok")
            .expect("Missing tokenizer.tok");
        let tokenizer: Tokenizer = bincode::deserialize(&tok_bytes)
            .expect("Tokenizer decode failed");

        let engine = Engine::load("toy_model.tmod");

        loop {
            print!("You: ");
            stdout().flush().unwrap();

            let mut input = String::new();
            stdin().read_line(&mut input).unwrap();
            let input = input.trim();
            if input == "exit" { break; }

            // add user input
            self.history.push(' ');
            self.history.push_str(input);

            let reply = self.generate(&tokenizer, &engine);
            println!("AI: {}", reply);

            // store reply in history for continuity
            self.history.push(' ');
            self.history.push_str(&reply);
        }
    }

    fn generate(&self, tokenizer: &Tokenizer, engine: &Engine) -> String {
        let enc = tokenizer.encode(&self.history);
        if enc.len() < 2 {
            return "…".to_string();
        }

        // last TWO tokens for 3-gram
        let mut a = enc[enc.len() - 2] as usize;
        let mut b = enc[enc.len() - 1] as usize;

        let mut out = Vec::new();

        // generate 12 tokens
        for _ in 0..12 {
            let c = engine.next_token(a, b);
            out.push(c as u32);

            // shift window
            a = b;
            b = c;
        }

        tokenizer.decode(out)
    }
}
