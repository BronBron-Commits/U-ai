use crate::model::{MAX_SEQ, D_MODEL};

pub fn attention_scores_forward(q: &[f32], k: &[f32], out: &mut [f32]) {
    let scale = 1.0 / (D_MODEL as f32).sqrt();
    for t in 0..MAX_SEQ {
        for u in 0..MAX_SEQ {
            let mut s = 0.0;
            for d in 0..D_MODEL {
                s += q[t*D_MODEL + d] * k[u*D_MODEL + d];
            }
            out[t*MAX_SEQ + u] = s * scale;
        }
    }
}

pub fn attention_softmax_forward(scores: &mut [f32], probs: &mut [f32]) {
    for t in 0..MAX_SEQ {
        for u in 0..MAX_SEQ {
            if u > t {
                scores[t*MAX_SEQ + u] = f32::NEG_INFINITY;
            }
        }
        let mut maxv = f32::NEG_INFINITY;
        for u in 0..MAX_SEQ {
            maxv = maxv.max(scores[t*MAX_SEQ + u]);
        }
        let mut sum = 0.0;
        for u in 0..MAX_SEQ {
            let v = (scores[t*MAX_SEQ + u] - maxv).exp();
            probs[t*MAX_SEQ + u] = v;
            sum += v;
        }
        for u in 0..MAX_SEQ {
            probs[t*MAX_SEQ + u] /= sum;
        }
    }
}

pub fn attention_apply_forward(probs: &[f32], v: &[f32], out: &mut [f32]) {
    for t in 0..MAX_SEQ {
        for d in 0..D_MODEL {
            let mut s = 0.0;
            for u in 0..MAX_SEQ {
                s += probs[t*MAX_SEQ + u] * v[u*D_MODEL + d];
            }
            out[t*D_MODEL + d] = s;
        }
    }
}
