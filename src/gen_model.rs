use rand::Rng;
use std::fs::File;
use std::io::Write;

pub fn generate() {
    const VOCAB: usize = 32;
    const D_MODEL: usize = 32;
    const D_FF: usize = 64;
    const MAX_TOKENS: usize = 128;

    let vocab_list: [&str; VOCAB] = [
        "hello","world","rust","ai","cat","dog","yes","no",
        "red","blue","sun","moon","good","bad","up","down",
        "left","right","star","end","why","how","when","who",
        "what","where","big","small","hot","cold","life","code",
    ];

    let mut f = File::create("toy_transformer.tmod").unwrap();

    // Header identifier + dims
    f.write_all(b"TMOD2").unwrap();
    f.write_all(&(VOCAB as u32).to_le_bytes()).unwrap();
    f.write_all(&(D_MODEL as u32).to_le_bytes()).unwrap();
    f.write_all(&(D_FF as u32).to_le_bytes()).unwrap();
    f.write_all(&(MAX_TOKENS as u32).to_le_bytes()).unwrap();

    let mut rng = rand::thread_rng();

    // write_mat must be mutable because it captures mutable rng
    let mut write_mat = |f: &mut File, rows: usize, cols: usize| {
        for _ in 0..rows {
            for _ in 0..cols {
                let v: f32 = rng.gen_range(-0.1..0.1);
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    };

    // Token embeddings
    write_mat(&mut f, VOCAB, D_MODEL);

    // Positional embeddings
    write_mat(&mut f, MAX_TOKENS, D_MODEL);

    // QKV: [D_MODEL x (3*D_MODEL)]
    write_mat(&mut f, D_MODEL, 3 * D_MODEL);

    // Output projection
    write_mat(&mut f, D_MODEL, D_MODEL);

    // FFN: W1 and W2
    write_mat(&mut f, D_MODEL, D_FF);
    write_mat(&mut f, D_FF, D_MODEL);
}
