use crate::entropy_sampler::get_rng;
use crate::tokenizer::Tokenizer;
use crate::llm_engine::LlmEngine;
use std::io::{stdin, stdout, Write};

pub struct ChatSession {
    pub history: String,
    pub max_context: usize,
}

impl ChatSession {
    pub fn new(max_context: usize) -> Self {
        ChatSession {
            history: String::new(),
            max_context,
        }
    }

    fn trim_context(&mut self) {
        if self.history.len() > self.max_context {
            let excess = self.history.len() - self.max_context;
            self.history.drain(0..excess);
        }
    }

    pub fn run(&mut self) {
        let tokenizer = Tokenizer::new("dataset.txt");
        let mut engine = LlmEngine::load("model.gguf").expect("Failed to load model");

        loop {
            print!("You: ");
            stdout().flush().unwrap();

            let mut input = String::new();
            stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input == "exit" { break; }

            // add user prompt
            self.history.push_str("\nUser: ");
            self.history.push_str(input);

            self.trim_context();

            // run inference
            let output = self.generate(&mut engine, &tokenizer);

            println!("AI: {}", output);

            // append model reply to history
            self.history.push_str("\nAI: ");
            self.history.push_str(&output);

            self.trim_context();
        }
    }

    fn generate(&self, engine: &mut LlmEngine, tokenizer: &Tokenizer) -> String {
        let mut rng = get_rng();
        let encoded = tokenizer.encode(&self.history);

        // model forward
        let logits = engine.forward(&encoded);

        // entropy-based sampling
        let probs = crate::entropy::sampler::softmax(&logits);
        let mut cumulative = 0.0;
        let choice: f32 = rng.gen();

        let mut next_token_idx = probs.len() - 1;
        for (i, p) in probs.iter().enumerate() {
            cumulative += *p;
            if choice < cumulative {
                next_token_idx = i;
                break;
            }
        }

        tokenizer.decode(vec![next_token_idx])
    }
}
