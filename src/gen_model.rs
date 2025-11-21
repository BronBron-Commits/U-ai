use rand::Rng;
use std::fs::File;
use std::io::Write;

pub const VOCAB: usize = 32;
pub const D_MODEL: usize = 64;
pub const D_FF: usize = 128;
pub const MAX_TOKENS: usize = 128;
pub const HEADS: usize = 4;

pub fn generate() {
    let mut f = File::create("toy_transformer.tmod").unwrap();

    // Magic header for this format
    f.write_all(b"TMOD3").unwrap();

    let write_u32 = |f: &mut File, x: usize| {
        f.write_all(&(x as u32).to_le_bytes()).unwrap();
    };

    write_u32(&mut f, VOCAB);
    write_u32(&mut f, D_MODEL);
    write_u32(&mut f, D_FF);
    write_u32(&mut f, MAX_TOKENS);
    write_u32(&mut f, HEADS);

    let mut rng = rand::thread_rng();

    // matrix writer must be mutable because RNG mutates
    let mut write_mat = |f: &mut File, rows: usize, cols: usize| {
        for _ in 0..rows {
            for _ in 0..cols {
                let v: f32 = rng.gen_range(-0.1..0.1);
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    };

    // token + positional + qkv + out + FF
    write_mat(&mut f, VOCAB, D_MODEL);
    write_mat(&mut f, MAX_TOKENS, D_MODEL);
    write_mat(&mut f, D_MODEL, 3 * D_MODEL);
    write_mat(&mut f, D_MODEL, D_MODEL);
    write_mat(&mut f, D_MODEL, D_FF);
    write_mat(&mut f, D_FF, D_MODEL);

    // LayerNorm gain/bias (four vectors)
    let mut write_vec = |f: &mut File, len: usize| {
        for _ in 0..len {
            let v: f32 = rng.gen_range(0.95..1.05);
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    };

    write_vec(&mut f, D_MODEL); // att_gain
    write_vec(&mut f, D_MODEL); // att_bias
    write_vec(&mut f, D_MODEL); // ffn_gain
    write_vec(&mut f, D_MODEL); // ffn_bias
}
