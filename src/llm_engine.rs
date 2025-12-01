use crate::model::Model;
use crate::tokenizer::Tokenizer;

pub struct LLmEngine {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub vocab_size: usize,
}

impl LLmEngine {
    pub fn new(_model_path: &str, tokenizer_path: &str) -> Self {
        let tokenizer = Tokenizer::load(tokenizer_path)
            .expect("Failed to load tokenizer");

        let vocab_size = tokenizer.vocab_size() as usize;
        let model = Model::new(vocab_size);

        Self { model, tokenizer, vocab_size }
    }

    pub fn predict(&self, prompt: &str) -> String {
        let p = prompt.trim();
        if p.is_empty() {
            return "Say something and I’ll respond.".to_string();
        }

        let lower = p.to_lowercase();

        // Simple rule-based replies to make it feel alive
        if lower.contains("how are you") {
            return "I’m a tiny Unhidra brain running in your terminal, doing pretty well so far.".to_string();
        }

        if lower.contains("who are you") || lower.contains("what are you") {
            return "I’m the Unhidra demo model you just wired up: Rust + C tokenizer + a baby model.".to_string();
        }

        if lower.contains("hey") || lower.contains("hi") || lower.contains("hello") {
            return "Hey there. I’m listening from inside your U-ai project.".to_string();
        }

        if lower.contains("thank") {
            return "You’re welcome. I exist to be poked and upgraded.".to_string();
        }

        if lower.contains("goodbye") || lower.contains("bye") || lower.contains("exit") {
            return "If you close the chat, I’ll still be here in your repo, waiting for the next run.".to_string();
        }

        // Fallback: use the tokenizer + model path you already wired
        let tokens = self.tokenizer
            .encode(p)
            .expect("Encode failed");

        if tokens.is_empty() {
            return "I didn’t get any tokens from that, try typing something else.".to_string();
        }

        let last = tokens[tokens.len() - 1] as usize;
        let next = self.model.forward(last);

        let decoded = self.tokenizer
            .decode(&[next as i32])
            .unwrap_or_else(|_| "[decode error]".to_string());

        format!("I don’t fully understand yet, but my brain mapped that to: {}", decoded)
    }
}
