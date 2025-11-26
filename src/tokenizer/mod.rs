use std::collections::HashMap;
use std::fs;

pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: Vec<String>,
}

impl Tokenizer {
    pub fn new(path: &str) -> Self {
        let text = fs::read_to_string(path).expect("dataset read failed");

        let mut vocab = HashMap::new();
        let mut inv_vocab = Vec::new();

        for word in text.split_whitespace() {
            if !vocab.contains_key(word) {
                let idx = vocab.len();
                vocab.insert(word.to_string(), idx);
                inv_vocab.push(word.to_string());
            }
        }

        Tokenizer { vocab, inv_vocab }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|w| *self.vocab.get(w).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, ids: Vec<usize>) -> String {
        ids.into_iter()
            .map(|i| {
                if i < self.inv_vocab.len() {
                    self.inv_vocab[i].clone()
                } else {
                    "<UNK>".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_len(&self) -> usize {
        self.inv_vocab.len()
    }
}
