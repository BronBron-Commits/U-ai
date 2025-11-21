use crate::model::*;
use super::{q::q_proj, k::k_proj, v::v_proj, scores::attn_scores, softmax::attn_softmax};

pub fn attn_layer(w: &LayerWeights, x: &[f32]) -> Vec<f32> {
    let q = q_proj(w, x);
    let k = k_proj(w, x);
    let v = v_proj(w, x);

    let scores = attn_scores(&q, &k, 1);
    let weights = attn_softmax(&scores);

    let mut out = vec![0.0; D_MODEL];
    for i in 0..D_MODEL {
        out[i] = weights[0] * v[i]; // 1-token toy model
    }

    // apply final linear projection (o)
    let mut proj = vec![0.0; D_MODEL];
    lin(&mut proj, &out, &w.o.w);
    proj
}

fn lin(out: &mut [f32], inp: &[f32], w: &[f32]) {
    let r = out.len();
    let c = inp.len();
    for i in 0..r {
        out[i] = 0.0;
        for j in 0..c {
            out[i] += inp[j] * w[i * c + j];
        }
    }
}
