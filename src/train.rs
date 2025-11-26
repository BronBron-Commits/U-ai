use std::fs;
use crate::tokenizer::Tokenizer;

pub fn train() {
    println!("Training conversational model…");

    // Merge all corpus text
    let mut merged = String::new();

    // 1. Base dataset
    if let Ok(base) = fs::read_to_string("dataset.txt") {
        merged.push_str(&base);
        merged.push('\n');
    }

    // 2. All corpora/*.txt
    if let Ok(entries) = fs::read_dir("corpora") {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Ok(t) = fs::read_to_string(&p) {
                    merged.push_str(&t);
                    merged.push('\n');
                }
            }
        }
    }

    // 3. Synthetic training patterns to teach THINK bubbles
    let synthetic = r#"
<USER> How does the system work?
<THINK>
Selected modules:
- modules/reasoning/logical_flow.txt

Reasoning summary:
User question matched “system”, “work”, “reasoning”
</THINK>
<AI> The system processes tasks by evaluating structure and applying logical flow.

<USER> Explain planning.
<THINK>
Selected modules:
- modules/general/planning.txt
</THINK>
<AI> Planning involves defining the goal, listing tasks, assigning time, and refining the steps.
"#;

    merged.push_str(synthetic);

    // Build tokenizer
    let tokenizer = Tokenizer::new_from_text(&merged);
    let tokens = tokenizer.encode(&merged);
    let vocab = tokenizer.vocab_len();

    // Save tokenizer
    let tok_bytes = bincode::serialize(&tokenizer).unwrap();
    fs::write("tokenizer.tok", tok_bytes).unwrap();

    // Build 2-gram model
    let mut table = vec![vec![0.0f32; vocab]; vocab];

    for win in tokens.windows(2) {
        let a = win[0] as usize;
        let b = win[1] as usize;
        table[a][b] += 1.0;
    }

    // Normalize
    for a in 0..vocab {
        let sum: f32 = table[a].iter().sum();
        if sum > 0.0 {
            for v in table[a].iter_mut() {
                *v /= sum;
            }
        }
    }

    // Save model
    let bytes = bincode::serialize(&table).unwrap();
    fs::write("toy_model.tmod", bytes).unwrap();

    println!("Model + tokenizer saved.");
}
