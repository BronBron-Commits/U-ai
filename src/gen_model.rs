use rand::Rng;
use std::fs::File;
use std::io::Write;

pub const D_MODEL: usize = 64;
pub const HEADS: usize = 4;
pub const LAYERS: usize = 2;
pub const D_FF: usize = 256;

pub const VOCAB: usize = 128;
pub const MAX_SEQ: usize = 128;

pub fn generate() {
    let mut f = File::create("toy_transformer.tmod").unwrap();
    f.write_all(b"TMOD3").unwrap();

    let write_u32 = |f: &mut File, x: usize| {
        f.write_all(&(x as u32).to_le_bytes()).unwrap();
    };

    write_u32(&mut f, VOCAB);
    write_u32(&mut f, D_MODEL);
    write_u32(&mut f, D_FF);
    write_u32(&mut f, MAX_SEQ);
    write_u32(&mut f, HEADS);
    write_u32(&mut f, LAYERS);

    let mut write_mat = |f: &mut File, rows: usize, cols: usize| {
        let mut rng = rand::thread_rng();
        for _ in 0..rows {
            for _ in 0..cols {
                let v: f32 = rng.gen_range(-0.1..0.1);
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    };

    let mut write_vec = |f: &mut File, len: usize| {
        let mut rng = rand::thread_rng();
        for _ in 0..len {
            let v: f32 = rng.gen_range(0.95..1.05);
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    };

    write_mat(&mut f, VOCAB, D_MODEL);
    write_mat(&mut f, MAX_SEQ, D_MODEL);

    for _ in 0..LAYERS {
        write_mat(&mut f, D_MODEL, D_MODEL);
        write_mat(&mut f, D_MODEL, D_MODEL);
        write_mat(&mut f, D_MODEL, D_MODEL);
        write_mat(&mut f, D_MODEL, D_MODEL);

        write_vec(&mut f, D_MODEL);
        write_vec(&mut f, D_MODEL);
        write_vec(&mut f, D_MODEL);
        write_vec(&mut f, D_MODEL);

        write_mat(&mut f, D_MODEL, D_FF);
        write_mat(&mut f, D_FF, D_MODEL);

        write_vec(&mut f, D_MODEL);
        write_vec(&mut f, D_MODEL);
    }
}
