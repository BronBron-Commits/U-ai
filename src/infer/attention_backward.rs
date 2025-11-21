use crate::model::{MAX_SEQ, D_MODEL};

pub fn attention_apply_backward(
    probs: &[f32],
    v: &[f32],
    grad_out: &[f32],
    dprobs: &mut [f32],
    dv: &mut [f32],
) {
    for t in 0..MAX_SEQ {
        for d in 0..D_MODEL {
            let go = grad_out[t*D_MODEL + d];
            for u in 0..MAX_SEQ {
                dprobs[t*MAX_SEQ + u] += go * v[u*D_MODEL + d];
                dv[u*D_MODEL + d] += go * probs[t*MAX_SEQ + u];
            }
        }
    }
}

pub fn softmax_backward(
    probs: &[f32],
    grad: &mut [f32],
) {
    for t in 0..MAX_SEQ {
        let mut dot = 0.0;
        for u in 0..MAX_SEQ {
            dot += grad[t*MAX_SEQ + u] * probs[t*MAX_SEQ + u];
        }
        for u in 0..MAX_SEQ {
            grad[t*MAX_SEQ + u] = probs[t*MAX_SEQ + u] * (grad[t*MAX_SEQ + u] - dot);
        }
    }
}

pub fn attention_scores_backward(
    q: &[f32],
    k: &[f32],
    dscore: &[f32],
    dq: &mut [f32],
    dk: &mut [f32],
) {
    let scale = 1.0 / (D_MODEL as f32).sqrt();
    for t in 0..MAX_SEQ {
        for u in 0..MAX_SEQ {
            let go = dscore[t*MAX_SEQ + u] * scale;
            for d in 0..D_MODEL {
                dq[t*D_MODEL + d] += go * k[u*D_MODEL + d];
                dk[u*D_MODEL + d] += go * q[t*D_MODEL + d];
            }
        }
    }
}
