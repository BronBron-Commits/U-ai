pub const D_MODEL: usize = 64;
pub const D_FF: usize = 256;
pub const VOCAB: usize = 256;
pub const MAX_SEQ: usize = 32;
pub const LAYERS: usize = 4;

use rand::Rng;

pub struct Param {
    pub w: Vec<f32>,
    pub grad: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl Param {
    pub fn new(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 0.02;        // small random init
        let mut w = Vec::with_capacity(size);
        for _ in 0..size {
            w.push(rng.gen_range(-scale..scale));
        }

        Param {
            w,
            grad: vec![0.0; size],
            m: vec![0.0; size],
            v: vec![0.0; size],
        }
    }
}

pub struct LayerWeights {
    pub q: Param,
    pub k: Param,
    pub v: Param,
    pub o: Param,
    pub ff1: Param,
    pub ff2: Param,
}

pub struct Model {
    pub token_emb: Param,
    pub pos_emb: Param,
    pub layers: Vec<LayerWeights>,
    pub final_proj: Param,
}
