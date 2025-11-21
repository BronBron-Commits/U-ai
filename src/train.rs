use std::fs;

pub struct CharDataset {
    pub chars: Vec<char>,
    pub stoi: std::collections::HashMap<char, usize>,
    pub itos: Vec<char>,
    pub data: Vec<usize>,
}

impl CharDataset {
    pub fn load(path: &str) -> Self {
        let text = fs::read_to_string(path).expect("Failed to read dataset file");

        // unique characters
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut stoi = std::collections::HashMap::new();
        for (i, ch) in chars.iter().enumerate() {
            stoi.insert(*ch, i);
        }

        let itos = chars.clone();

        let data: Vec<usize> = text
            .chars()
            .filter_map(|c| stoi.get(&c).cloned())
            .collect();

        Self { chars, stoi, itos, data }
    }

    // Create mini-batches for training
    pub fn get_batch(&self, batch_size: usize, seq_len: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();

        let data_len = self.data.len();

        for _ in 0..batch_size {
            let idx = rand::random::<usize>() % (data_len - seq_len - 1);
            let x = self.data[idx..idx + seq_len].to_vec();
            let y = self.data[idx + 1..idx + seq_len + 1].to_vec();
            xs.push(x);
            ys.push(y);
        }

        (xs, ys)
    }
}
