use crate::model::Model;
use crate::tokenizer::Tokenizer;
use crate::entropy::get_entropy_byte;

pub struct LLmEngine {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub vocab_size: usize,
}

impl LLmEngine {
    pub fn new(_model_path: &str, tokenizer_path: &str, _vocab_path: &str) -> Self {
        let tokenizer = Tokenizer::load(tokenizer_path).expect("Failed to load tokenizer");
        let vocab_size = tokenizer.vocab_size() as usize;
        let model = Model::new(vocab_size);
        Self { model, tokenizer, vocab_size }
    }

    pub fn predict(&self, prompt: &str) -> String {
        let p = prompt.trim().to_lowercase();
        if p.is_empty() {
            return "Say something and I’ll respond.".to_string();
        }

        if p == "how are you" || p == "how are you?" {
            return "Doing well, entropy flowing strong.".into();
        }
        if p.contains("who are you") {
            return "I’m the Unhidra prototype — now powered by live entropy.".into();
        }
        if ["hey", "hi", "hello", "yo"].contains(&p.as_str()) {
            return "Hey there. I’m listening from inside your U-ai project.".into();
        }

        // Entropy-driven fallback
        let id = self.model.forward_with_entropy(0);
        let e = get_entropy_byte();
        let responses = [
            "Something feels electric today.",
            "Entropy stirs in the wires…",
            "Hard to say — randomness rules me now.",
            "I'm sensing heat from the chaos feed.",
            "Static whispers, patterns shifting.",
            "I saw a flicker — maybe that was meaning?",
            "I'm thinking, if you can call it that.",
        ];
        let index = (id as usize + e as usize) % responses.len();
        responses[index].to_string()
    }
}
