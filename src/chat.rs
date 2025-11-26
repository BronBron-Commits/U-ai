use std::io::{stdin, stdout, Write};
use crate::tokenizer::Tokenizer;
use crate::llm_engine::Engine;

pub struct ChatSession {
    pub history: String,
    pub max_context: usize,
}

impl ChatSession {
    pub fn new(max_context: usize) -> Self {
        ChatSession { history: String::new(), max_context }
    }

    pub fn run(&mut self) {
        let tokenizer = Tokenizer::new_from_text(&std::fs::read_to_string("dataset.txt").unwrap_or_default());
        let engine = Engine::load("toy_model.tmod");

        loop {
            print!("You: ");
            stdout().flush().unwrap();

            let mut input = String::new();
            stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input == "exit" { break; }

            self.history.push_str(" ");
            self.history.push_str(input);

            let reply = self.generate(&tokenizer, &engine);
            println!("AI: {}", reply);

            self.history.push_str(" ");
            self.history.push_str(&reply);
        }
    }

    fn generate(&self, tokenizer: &Tokenizer, engine: &Engine) -> String {
        let enc = tokenizer.encode(&self.history);
        if enc.is_empty() { return "…".to_string(); }

        let mut prev = enc[enc.len() - 1] as usize;
        let mut out = Vec::new();

        for _ in 0..15 {
            let next = engine.next_token(prev);
            out.push(next as u32);
            prev = next;
        }

        tokenizer.decode(out)
    }
}
