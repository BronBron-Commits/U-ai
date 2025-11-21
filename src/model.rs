use std::fs::File;
use std::io::{Read, Write};

pub const VOCAB: usize = 128;
pub const D_MODEL: usize = 64;
pub const D_FF: usize = 256;
pub const HEADS: usize = 4;
pub const LAYERS: usize = 2;
pub const MAX_SEQ: usize = 128;

pub struct Model {
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
}

impl Model {
    pub fn load(path: &str) -> Self {
        let mut f = File::open(path).unwrap();
        let mut magic = [0u8; 5];
        f.read_exact(&mut magic).unwrap();

        assert!(&magic == b"TMOD3");

        let mut buf = [0u8; 4];
        f.read_exact(&mut buf).unwrap();
        let vocab = u32::from_le_bytes(buf) as usize;

        f.read_exact(&mut buf).unwrap();
        let d_model = u32::from_le_bytes(buf) as usize;

        f.read_exact(&mut buf).unwrap();
        let _d_ff = u32::from_le_bytes(buf) as usize;

        f.read_exact(&mut buf).unwrap();
        let _max_seq = u32::from_le_bytes(buf) as usize;

        f.read_exact(&mut buf).unwrap();
        let _heads = u32::from_le_bytes(buf) as usize;

        f.read_exact(&mut buf).unwrap();
        let _layers = u32::from_le_bytes(buf) as usize;

        let mut token_emb = vec![0.0; vocab * d_model];
        let mut pos_emb = vec![0.0; MAX_SEQ * d_model];

        let token_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                token_emb.as_mut_ptr() as *mut u8,
                token_emb.len() * 4,
            )
        };

        let pos_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                pos_emb.as_mut_ptr() as *mut u8,
                pos_emb.len() * 4,
            )
        };

        f.read_exact(token_bytes).unwrap();
        f.read_exact(pos_bytes).unwrap();

        Model { token_emb, pos_emb }
    }
}
