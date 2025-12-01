use std::fs;

pub struct Vocab {
    pub tokens: Vec<String>,
}

impl Vocab {
    pub fn load(path: &str) -> Result<Self, String> {
        let data = fs::read_to_string(path).map_err(|_| "Failed to read vocab file")?;
        let tokens = data
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect();
        Ok(Self { tokens })
    }

    pub fn decode_id(&self, id: usize) -> String {
        if id < self.tokens.len() {
            self.tokens[id].clone()
        } else {
            format!("<unk:{}>", id)
        }
    }
}
