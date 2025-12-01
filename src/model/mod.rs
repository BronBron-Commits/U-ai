use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl Model {
    pub fn new(vocab: usize) -> Self {
        Self {
            w: vec![0.0; vocab],
            b: vec![0.0; vocab],
        }
    }

    pub fn load_tmod(path: &str) -> Result<Self, String> {
        let data = fs::read_to_string(path).map_err(|_| "Failed to read .tmod")?;
        let mut lines = data.lines();

        if lines.next() != Some("UAI_TMOD_V1") {
            return Err("Invalid TMOD header".into());
        }

        let weights_line = lines.next().unwrap_or("");
        let weights_count: usize = weights_line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
        let weights: Vec<f32> = lines
            .next()
            .unwrap_or("")
            .split_whitespace()
            .take(weights_count)
            .filter_map(|v| v.parse::<f32>().ok())
            .collect();

        let biases_line = lines.next().unwrap_or("");
        let biases_count: usize = biases_line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
        let biases: Vec<f32> = lines
            .next()
            .unwrap_or("")
            .split_whitespace()
            .take(biases_count)
            .filter_map(|v| v.parse::<f32>().ok())
            .collect();

        Ok(Self { w: weights, b: biases })
    }

    pub fn forward(&self, input_token: usize) -> usize {
        use rand::Rng;

        if self.w.is_empty() || self.b.is_empty() {
            return input_token;
        }

        // lightweight softmax over bias values for demonstration
        let max_b = self.b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = self.b.iter().map(|v| (v - max_b).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();

        // pick weighted random index
        let mut rng = rand::thread_rng();
        let mut r: f32 = rng.gen::<f32>() * sum_exp;
        for (i, e) in exps.iter().enumerate() {
            if r < *e {
                return i;
            }
            r -= e;
        }

        input_token % self.b.len()
    }
}

use crate::entropy::get_entropy_byte;

impl Model {
    /// Use entropy to pick a pseudo-random index weighted by model bias.
    pub fn sample_with_entropy(&self) -> usize {
        if self.w.is_empty() || self.b.is_empty() {
            return 0;
        }

        let entropy_val = get_entropy_byte() as usize;
        let idx = entropy_val % self.b.len();
        idx
    }

    /// Forward method that leverages entropy to drive selection
    pub fn forward_with_entropy(&self, _input: usize) -> usize {
        self.sample_with_entropy()
    }
}
