use bincode;

pub struct Engine {
    table: Vec<Vec<f32>>,
}

impl Engine {
    pub fn load(path: &str) -> Self {
        let bytes = std::fs::read(path).expect("model load failed");
        let table: Vec<Vec<f32>> = bincode::deserialize(&bytes).unwrap();
        Engine { table }
    }

    pub fn next_token(&self, prev: usize) -> usize {
        if prev >= self.table.len() { return 0; }
        let row = &self.table[prev];

        let sum: f32 = row.iter().sum();
        if sum == 0.0 { return 0; }

        let mut acc = 0.0;
        let mut choice = rand::random::<f32>() * sum;

        for (idx, val) in row.iter().enumerate() {
            acc += *val;
            if acc >= choice {
                return idx;
            }
        }

        0
    }
}
