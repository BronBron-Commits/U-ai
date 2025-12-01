use serde::{Serialize, Deserialize};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl Model {
    pub fn new(vocab: usize) -> Self {
        Self { w: vec![0.0; vocab], b: vec![0.0; vocab] }
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

    pub fn forward(&self, input_token: usize) -> usize {
        // Use current time to mix things up a bit
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as usize;
        ((input_token.wrapping_mul(31) ^ now) % self.w.len()).max(1)
    }
}
