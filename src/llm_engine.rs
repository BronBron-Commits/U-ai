use std::fs;
use serde::{Serialize, Deserialize};
use rand::Rng;

#[derive(Serialize, Deserialize)]
pub struct Sparse3Gram {
    pub cube: std::collections::HashMap<(u32, u32), Vec<(u32, f32)>>, 
}

pub struct Engine {
    pub model: Sparse3Gram,
}

impl Engine {
    pub fn load(path: &str) -> Self {
        let bytes = fs::read(path).expect("Missing model file");
        let model: Sparse3Gram = bincode::deserialize(&bytes)
            .expect("Failed to load sparse 3-gram");

        Engine { model }
    }

    pub fn next_token(&self, a: usize, b: usize) -> usize {
        let key = (a as u32, b as u32);

        let row = match self.model.cube.get(&key) {
            Some(r) => r,
            None => return 0, // fallback
        };

        let mut rng = rand::thread_rng();
        let mut choice = rng.gen::<f32>();

        for (token, prob) in row {
            if choice <= *prob {
                return *token as usize;
            }
            choice -= *prob;
        }

        row.last().map(|(t,_)| *t as usize).unwrap_or(0)
    }
}
