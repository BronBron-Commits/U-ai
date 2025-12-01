use crate::model::Model;
use crate::tokenizer::Tokenizer;

pub struct Engine {
    model: Model,
    tokenizer: Tokenizer,
}

impl Engine {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Self {
        let tokenizer = Tokenizer::load(tokenizer_path)
            .expect("Failed to load tokenizer");

        let vocab = tokenizer.vocab_size().try_into().unwrap();
        let model = Model::new(vocab);

        Self { model, tokenizer }
    }

    pub fn predict(&self, text: &str) -> String {
        let tokens = self.tokenizer.encode(text).unwrap();
        if tokens.is_empty() {
            return String::new();
        }
        let last = tokens[tokens.len() - 1] as usize;
        let next = last % self.model.w.len();
        self.tokenizer.decode(&[next.try_into().unwrap()])
            .expect("decode failed")
    }
}
