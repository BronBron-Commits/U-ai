use std::collections::HashMap;
use std::fs;
use crate::tokenizer::Tokenizer;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Sparse3Gram {
    pub cube: HashMap<(u32, u32), Vec<(u32, f32)>>, 
}

pub fn train() {
    println!("Training sparse 3-gram model…");

    // Merge all corpora into one string
    let mut merged = String::new();

    if let Ok(t) = fs::read_to_string("dataset.txt") {
        merged.push_str(&t);
        merged.push('\n');
    }

    if let Ok(entries) = fs::read_dir("corpora") {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Ok(t) = fs::read_to_string(&p) {
                    merged.push_str(&t);
                    merged.push('\n');
                }
            }
        }
    }

    // Tokenizer + encode
    let tokenizer = Tokenizer::new_from_text(&merged);
    let tokens = tokenizer.encode(&merged);

    // Save tokenizer
    let tb = bincode::serialize(&tokenizer).unwrap();
    fs::write("tokenizer.tok", tb).unwrap();

    let mut cube: HashMap<(u32, u32), HashMap<u32, f32>> = HashMap::new();

    // Collect counts: (a,b)->c
    for win in tokens.windows(3) {
        let a = win[0];
        let b = win[1];
        let c = win[2];

        cube.entry((a, b))
            .or_insert_with(HashMap::new)
            .entry(c)
            .and_modify(|v| *v += 1.0)
            .or_insert(1.0);
    }

    // Normalize into Vec<(token, prob)>
    let mut final_cube: HashMap<(u32, u32), Vec<(u32, f32)>> = HashMap::new();

    for ((a, b), row) in cube.into_iter() {
        let sum: f32 = row.values().sum();
        let mut vec_row = Vec::new();

        if sum > 0.0 {
            for (c, count) in row {
                vec_row.push((c, count / sum));
            }
        }

        final_cube.insert((a, b), vec_row);
    }

    // Save model
    let model = Sparse3Gram { cube: final_cube };
    let mb = bincode::serialize(&model).unwrap();
    fs::write("toy_model.tmod", mb).unwrap();

    println!("Sparse 3-gram model saved.");
}
