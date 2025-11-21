use std::fs::File;
use std::io::{Read, Write};
use rand::Rng;

pub const D_MODEL: usize = 64;
pub const HEADS: usize = 4;
pub const LAYERS: usize = 2;
pub const D_FF: usize = 256;

pub const VOCAB: usize = 128;        // ASCII
pub const MAX_SEQ: usize = 128;      // training context size

// A tensor with gradients + Adam moments
#[derive(Clone)]
pub struct Param {
    pub w: Vec<f32>,
    pub grad: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl Param {
    pub fn new(size: usize, scale: f32) -> Self {
        let mut rng = rand::thread_rng();
        let mut w = Vec::with_capacity(size);
        for _ in 0..size {
            w.push(rng.gen_range(-scale..scale));
        }
        Self {
            w,
            grad: vec![0.0; size],
            m: vec![0.0; size],
            v: vec![0.0; size],
        }
    }
}

// Full transformer weights
pub struct Model {
    pub token_emb: Param,
    pub pos_emb: Param,

    pub q: Vec<Param>,
    pub k: Vec<Param>,
    pub v: Vec<Param>,
    pub o: Vec<Param>,

    pub ln1_gain: Vec<Param>,
    pub ln1_bias: Vec<Param>,

    pub ffc1: Vec<Param>,
    pub ffc2: Vec<Param>,

    pub ln2_gain: Vec<Param>,
    pub ln2_bias: Vec<Param>,
}

impl Model {
    pub fn new() -> Self {
        fn p(size: usize) -> Param { Param::new(size, 0.02) }

        let mut q = Vec::new();
        let mut k = Vec::new();
        let mut v = Vec::new();
        let mut o = Vec::new();
        let mut ln1_gain = Vec::new();
        let mut ln1_bias = Vec::new();
        let mut ffc1 = Vec::new();
        let mut ffc2 = Vec::new();
        let mut ln2_gain = Vec::new();
        let mut ln2_bias = Vec::new();

        for _ in 0..LAYERS {
            q.push(p(D_MODEL * D_MODEL));
            k.push(p(D_MODEL * D_MODEL));
            v.push(p(D_MODEL * D_MODEL));
            o.push(p(D_MODEL * D_MODEL));

            ln1_gain.push(p(D_MODEL));
            ln1_bias.push(p(D_MODEL));

            ffc1.push(p(D_MODEL * D_FF));
            ffc2.push(p(D_FF * D_MODEL));

            ln2_gain.push(p(D_MODEL));
            ln2_bias.push(p(D_MODEL));
        }

        Self {
            token_emb: p(VOCAB * D_MODEL),
            pos_emb: p(MAX_SEQ * D_MODEL),
            q,
            k,
            v,
            o,
            ln1_gain,
            ln1_bias,
            ffc1,
            ffc2,
            ln2_gain,
            ln2_bias,
        }
    }

    pub fn zero_grads(&mut self) {
        for t in self.all_params_mut() {
            for g in t.grad.iter_mut() {
                *g = 0.0;
            }
        }
    }

    pub fn all_params_mut(&mut self) -> Vec<&mut Param> {
        let mut v = Vec::new();
        v.push(&mut self.token_emb);
        v.push(&mut self.pos_emb);
        for i in 0..LAYERS {
            v.push(&mut self.q[i]);
            v.push(&mut self.k[i]);
            v.push(&mut self.v[i]);
            v.push(&mut self.o[i]);
            v.push(&mut self.ln1_gain[i]);
            v.push(&mut self.ln1_bias[i]);
            v.push(&mut self.ffc1[i]);
            v.push(&mut self.ffc2[i]);
            v.push(&mut self.ln2_gain[i]);
            v.push(&mut self.ln2_bias[i]);
        }
        v
    }

    pub fn save(&self, path: &str) {
        let mut f = File::create(path).unwrap();
        for t in self.all_params_readonly() {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    t.as_ptr() as *const u8,
                    t.len() * 4
                )
            };
            f.write_all(bytes).unwrap();
        }
    }

    pub fn all_params_readonly(&self) -> Vec<&[f32]> {
        let mut v = Vec::new();
        v.push(&self.token_emb.w);
        v.push(&self.pos_emb.w);
        for i in 0..LAYERS {
            v.push(&self.q[i].w);
            v.push(&self.k[i].w);
            v.push(&self.v[i].w);
            v.push(&self.o[i].w);
            v.push(&self.ln1_gain[i].w);
            v.push(&self.ln1_bias[i].w);
            v.push(&self.ffc1[i].w);
            v.push(&self.ffc2[i].w);
            v.push(&self.ln2_gain[i].w);
            v.push(&self.ln2_bias[i].w);
        }
        v
    }
}
