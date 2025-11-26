use crate::tokenizer::Tokenizer;
use rand::Rng;

pub fn train() {
    let tokenizer = Tokenizer::new("dataset.txt");
    let data = std::fs::read_to_string("dataset.txt").unwrap();
    let lines: Vec<&str> = data.lines().collect();
    let mut rng = rand::thread_rng();

    println!("Training tiny model...");

    let vocab = tokenizer.vocab_len();
    let mut table = vec![vec![0.0f32; vocab]; vocab];

    for step in 0..5000 {
        let line = lines[rng.gen_range(0..lines.len())];
        let toks = tokenizer.encode(line);

        for pair in toks.windows(2) {
            let a = pair[0];
            let b = pair[1];
            table[a][b] += 1.0;
        }

        if step % 500 == 0 {
            println!("step {}", step);
        }
    }

    let bytes = bincode::serialize(&table).unwrap();
    std::fs::write("toy_model.tmod", bytes).unwrap();
    println!("Training complete → toy_model.tmod written");
}
