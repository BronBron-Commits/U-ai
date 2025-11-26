use std::collections::HashMap;

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

        // Ensure <SEP> always exists
        if !token_to_id.contains_key("<SEP>") {
            let id = id_to_token.len() as u32;
            token_to_id.insert("<SEP>".to_string(), id);
            id_to_token.push("<SEP>".to_string());
        }

        Self { token_to_id, id_to_token }
    }

    // kept for backward compatibility
    pub fn new(_path: &str) -> Self {
        Self::new_from_text("")
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|w| *self.token_to_id.get(w).unwrap_or(&self.token_to_id["<SEP>"]))
            .collect()
    }

    pub fn decode(&self, ids: Vec<u32>) -> String {
        ids.iter()
            .filter_map(|id| self.id_to_token.get(*id as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_len(&self) -> usize {
        self.id_to_token.len()
    }
}
