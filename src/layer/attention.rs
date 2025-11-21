pub mod q;
pub mod k;
pub mod v;
pub mod scores;
pub mod softmax;
pub mod weighted;

use crate::model::*;
use q::q_proj;
use k::k_proj;
use v::v_proj;
use scores::attn_scores;
use softmax::attn_softmax;
use weighted::attn_weighted;

pub fn attn_layer(w: &LayerWeights, x: &[f32], seq_len: usize) -> Vec<f32> {
    let q = q_proj(w, x);
    let k = k_proj(w, x);
    let v = v_proj(w, x);

    let scores = attn_scores(&q, &k, seq_len);
    let weights = attn_softmax(&scores);
    attn_weighted(&v, &weights)
}
