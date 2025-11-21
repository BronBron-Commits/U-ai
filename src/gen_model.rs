use rand::Rng;
use std::fs::File;
use std::io::Write;

pub fn generate() {
    const VOCAB: usize = 16;
    const EMBED: usize = 8;
    const HIDDEN: usize = 8;

    let vocab: [&str; VOCAB] = [
        "hello", "world", "rust", "love",
        "cat", "dog", "good", "bad",
        "yes", "no", "red", "blue",
        "sun", "moon", "star", "end",
    ];

    let mut f = File::create("toy_model.tmod").unwrap();

    // header
    f.write_all(b"TMOD").unwrap();
    f.write_all(&(VOCAB as u32).to_le_bytes()).unwrap();
    f.write_all(&(EMBED as u32).to_le_bytes()).unwrap();
    f.write_all(&(HIDDEN as u32).to_le_bytes()).unwrap();

    let mut rng = rand::thread_rng();

    // embeddings
    for _ in 0..VOCAB {
        for _ in 0..EMBED {
            let v: f32 = rng.gen_range(-0.5..0.5);
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    // W1
    for _ in 0..EMBED {
        for _ in 0..HIDDEN {
            let v: f32 = rng.gen_range(-0.5..0.5);
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    // W2
    for _ in 0..HIDDEN {
        for _ in 0..VOCAB {
            let v: f32 = rng.gen_range(-0.5..0.5);
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}
