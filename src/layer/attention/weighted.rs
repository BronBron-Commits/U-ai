use crate::model::*;

pub fn attn_weighted(v: &[f32], weights: &[f32]) -> Vec<f32> {
    let w = weights[0];
    let mut out = vec![0.0; D_MODEL];
    for i in 0..D_MODEL {
        out[i] = v[i] * w;
    }
    out
}
