use crate::model::Model;
use crate::tokenizer::Tokenizer;

pub struct LLmEngine {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub vocab_size: usize,
}

impl LLmEngine {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Self {
        let tokenizer = Tokenizer::load(tokenizer_path)
            .expect("Failed to load tokenizer");

        let vocab_size = tokenizer.vocab_size() as usize;

        let model = Model::new(vocab_size);

        Self { model, tokenizer, vocab_size }
    }

    pub fn predict(&self, prompt: &str) -> String {
        let tokens = self.tokenizer.encode(prompt)
            .expect("Encode failed");

        if tokens.is_empty() {
            return String::new();
        }

        let last = tokens[tokens.len() - 1] as usize;

        let next = self.model.forward(last);

        self.tokenizer
            .decode(&[next as i32])
            .expect("Decode failed")
    }
}
