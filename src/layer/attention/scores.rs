use crate::model::*;

pub fn attn_scores(q: &[f32], k: &[f32], seq_len: usize) -> Vec<f32> {
    let mut score = 0.0;
    for i in 0..D_MODEL {
        score += q[i] * k[i];
    }

    if seq_len == 0 {
        score = f32::NEG_INFINITY;
    }

    vec![score]
}
