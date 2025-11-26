use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Tokenizer {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
}

impl Tokenizer {
    pub fn new_from_text(text: &str) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        for w in text.split_whitespace() {
            if !token_to_id.contains_key(w) {
                let id = id_to_token.len() as u32;
                token_to_id.insert(w.to_string(), id);
                id_to_token.push(w.to_string());
            }
        }

        if !token_to_id.contains_key("<UNK>") {
            let id = id_to_token.len() as u32;
            token_to_id.insert("<UNK>".to_string(), id);
            id_to_token.push("<UNK>".to_string());
        }

        Self { token_to_id, id_to_token }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|w| *self.token_to_id.get(w).unwrap_or(&self.token_to_id["<UNK>"]))
            .collect()
    }

    pub fn decode(&self, ids: Vec<u32>) -> String {
        ids.into_iter()
            .filter_map(|id| self.id_to_token.get(id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_len(&self) -> usize {
        self.id_to_token.len()
    }
}
