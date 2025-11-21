use rand::Rng;
use std::fs::File;
use std::io::Write;

pub fn generate() {
    const VOCAB: usize = 32;
    const D_MODEL: usize = 32;
    const D_FF: usize = 64;
    const MAX_TOKENS: usize = 128;
    const HEADS: usize = 4;

    let mut f = File::create("toy_transformer.tmod").unwrap();

    f.write_all(b"TMOD2").unwrap();
    f.write_all(&(VOCAB as u32).to_le_bytes()).unwrap();
    f.write_all(&(D_MODEL as u32).to_le_bytes()).unwrap();
    f.write_all(&(D_FF as u32).to_le_bytes()).unwrap();
    f.write_all(&(MAX_TOKENS as u32).to_le_bytes()).unwrap();
    f.write_all(&(HEADS as u32).to_le_bytes()).unwrap();

    let mut rng = rand::thread_rng();

    let mut write_mat = |rows: usize, cols: usize| {
        for _ in 0..rows {
            for _ in 0..cols {
                let v: f32 = rng.gen_range(-0.1..0.1);
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    };

    write_mat(VOCAB, D_MODEL);            // token embeddings  
    write_mat(MAX_TOKENS, D_MODEL);       // positional embeddings  

    write_mat(D_MODEL, 3 * D_MODEL);      // QKV projection  
    write_mat(D_MODEL, D_MODEL);          // output proj  

    write_mat(D_MODEL, D_FF);             // FFN W1  
    write_mat(D_FF, D_MODEL);             // FFN W2  
}
