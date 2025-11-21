use rand::Rng;
use crate::model::*;
pub mod attention;
pub mod ff;

use attention::attn_layer;
use ff::ff_layer;

pub fn forward_full(m: &mut Model, tokens: &[usize]) -> usize {
    let n = tokens.len();

    let mut x = vec![0.0; D_MODEL];
    for d in 0..D_MODEL {
        x[d] = m.token_emb.w[tokens[n-1] * D_MODEL + d]
             + m.pos_emb.w[(n-1) * D_MODEL + d];
    }

    for l in 0..LAYERS {
        let w = &m.layers[l];
        let a = attn_layer(w, &x, n);
        for i in 0..D_MODEL {
            x[i] += a[i];
        }
        let f = ff_layer(w, &x);
        for i in 0..D_MODEL {
            x[i] += f[i];
        }
    }

    let mut logits = vec![0.0; VOCAB];
    for v in 0..VOCAB {
        let off = v * D_MODEL;
        let mut acc = 0.0;
        for d in 0..D_MODEL {
            acc += x[d] * m.final_proj.w[off + d];
        }
        logits[v] = acc;
    }

    softmax_sample(&logits)
}

fn softmax_sample(log: &[f32]) -> usize {
    let mut max = f32::NEG_INFINITY;
    for &v in log {
        if v > max { max = v; }
    }

    let mut sum = 0.0;
    let mut probs = vec![0.0; log.len()];
    for i in 0..log.len() {
        let e = (log[i] - max).exp();
        probs[i] = e;
        sum += e;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();

    let mut acc = 0.0;
    for i in 0..probs.len() {
        acc += probs[i];
        if r < acc {
            return i;
        }
    }
    probs.len() - 1
}
