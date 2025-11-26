use std::fs;
use crate::tokenizer::Tokenizer;

pub fn train() {
    println!("Training 2-gram model (merged corpus)…");

    let mut merged = String::new();

    // 1. Dataset
    if let Ok(base) = fs::read_to_string("dataset.txt") {
        merged.push_str(&base);
        merged.push('\n');
    }

    // 2. Modules
    if let Ok(entries) = fs::read_dir("modules/github") {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Ok(content) = fs::read_to_string(&p) {
                    merged.push_str(&content);
                    merged.push('\n');
                }
            }
        }
    }

    // 3. English corpora
    if let Ok(entries) = fs::read_dir("corpora") {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Ok(content) = fs::read_to_string(&p) {
                    merged.push_str(&content);
                    merged.push('\n');
                }
            }
        }
    }

    // 4. Tokenizer
    let tokenizer = Tokenizer::new_from_text(&merged);
    let tokens = tokenizer.encode(&merged);
    let vocab = tokenizer.vocab_len();

    // 5. Bigram table
    let mut table = vec![vec![0.0f32; vocab]; vocab];

    for win in tokens.windows(2) {
        let a = win[0] as usize;
        let b = win[1] as usize;
        table[a][b] += 1.0;
    }

    // 6. Module boosting (keywords → module vocabulary)
    let mut module_words = Vec::<usize>::new();

    if let Ok(entries) = fs::read_dir("modules/github") {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Ok(content) = fs::read_to_string(&p) {
                    for w in content.split_whitespace() {
                        if let Some(&id) = tokenizer.token_to_id.get(w) {
                            module_words.push(id as usize);
                        }
                    }
                }
            }
        }
    }

    let triggers = [
        "project","projects","github","repo",
        "module","modules","chat","u-chat",
        "ai","system","kernel"
    ];

    for trig in triggers {
        if let Some(&tid) = tokenizer.token_to_id.get(trig) {
            let t = tid as usize;

            for &mw in &module_words {
                if t < vocab && mw < vocab {
                    table[t][mw] += 2.0;
                }
            }
        }
    }

    // 7. Normalize
    for a in 0..vocab {
        let sum: f32 = table[a].iter().sum();
        if sum > 0.0 {
            for v in table[a].iter_mut() {
                *v /= sum;
            }
        }
    }

    // 8. Save
    let bytes = bincode::serialize(&table).unwrap();
    fs::write("toy_model.tmod", bytes).unwrap();

    println!("Model training complete (2-gram).");
}
