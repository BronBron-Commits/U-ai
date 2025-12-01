use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl Model {
    pub fn new(vocab: usize) -> Self {
        // Tiny placeholder model
        Self {
            w: vec![0.0; vocab],
            b: vec![0.0; vocab],
        }
    }

    pub fn load(path: &str) -> Result<Self, ()> {
        let data = fs::read(path).map_err(|_| ())?;
        let model: Self = bincode::deserialize(&data).map_err(|_| ())?;
        Ok(model)
    }

    pub fn save(&self, path: &str) {
        let bytes = bincode::serialize(self).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}

impl Model {
    pub fn forward(&self, input_token: usize) -> usize {
        // Placeholder logic: just shift the token
        (input_token + 1) % self.w.len()
    }
}
