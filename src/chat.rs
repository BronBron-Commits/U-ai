use std::io::{stdin, stdout, Write};
use crate::tokenizer::Tokenizer;
use crate::llm_engine::Engine;
use crate::modules::{select_modules, load_module_texts};

pub struct ChatSession {
    pub history: String,
}

impl ChatSession {
    pub fn new() -> Self {
        ChatSession { history: String::new() }
    }

    pub fn run(&mut self) {
        // Load saved tokenizer
        let tok_bytes = std::fs::read("tokenizer.tok")
            .expect("Missing tokenizer.tok");
        let tokenizer: Tokenizer = bincode::deserialize(&tok_bytes)
            .expect("Tokenizer decode failed");

        // Load model
        let engine = Engine::load("toy_model.tmod");

        loop {
            print!("You: ");
            stdout().flush().unwrap();

            let mut input = String::new();
            stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input == "exit" {
                break;
            }

            // === MODULE SELECTION ===
            let hit_paths = select_modules(input);
            let module_texts = load_module_texts(&hit_paths);

            // Build THINK bubble
            let mut think = String::new();
            think.push_str("<THINK>\nSelected modules:\n");
            for p in &hit_paths {
                think.push_str(&format!("- {}\n", p));
            }
            think.push_str("\nReasoning summary: user msg → keyword match → module text injected\n</THINK>\n");

            // Add THINK bubble + modules into the history
            if !module_texts.is_empty() {
                self.history.push(' ');
                self.history.push_str(&think);

                for m in module_texts {
                    self.history.push(' ');
                    self.history.push_str(&m);
                }
            }

            // finally add the user message
            self.history.push(' ');
            self.history.push_str(input);

            // === GENERATION ===
            let reply = self.generate(&tokenizer, &engine);
            println!("AI: {}", reply);

            self.history.push(' ');
            self.history.push_str(&reply);
        }
    }

    fn generate(&self, tokenizer: &Tokenizer, engine: &Engine) -> String {
        let enc = tokenizer.encode(&self.history);
        if enc.is_empty() {
            return "…".to_string();
        }

        let mut cur = enc[enc.len() - 1] as usize;
        let mut out = Vec::new();

        for _ in 0..12 {
            cur = engine.next_token(cur);
            out.push(cur as u32);
        }

        tokenizer.decode(out)
    }
}
