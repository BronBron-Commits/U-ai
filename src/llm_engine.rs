use std::fs;

pub struct Engine {
    pub table: Vec<Vec<f32>>,   // bigram: prev → distribution over next
}

impl Engine {
    pub fn load(path: &str) -> Self {
        let bytes = fs::read(path).expect("Failed to read model file");

        let table: Vec<Vec<f32>> = match bincode::deserialize(&bytes) {
            Ok(t) => t,
            Err(e) => panic!("Failed to deserialize model: {:?}", e),
        };

        Engine { table }
    }

    pub fn next_token(&self, prev: usize) -> usize {
        if prev >= self.table.len() { return 0; }

        let row = &self.table[prev];
        let sum: f32 = row.iter().sum();

        if sum <= 0.0 {
            return 0;
        }

        let choice = rand::random::<f32>() * sum;
        let mut acc = 0.0;

        for (i, v) in row.iter().enumerate() {
            acc += *v;
            if acc >= choice {
                return i;
            }
        }

        row.len() - 1
    }
}
