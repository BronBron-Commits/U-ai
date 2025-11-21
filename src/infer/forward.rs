use crate::model::*;
use crate::infer::{
    layernorm_forward::layernorm_forward,
    attention_forward::{attention_scores_forward, attention_softmax_forward, attention_apply_forward},
    ff_forward::{ff1_forward, gelu_forward, ff2_forward},
    matmul_forward::matmul_forward,
};

pub struct Cache {
    pub x0: Vec<f32>,
    pub ln1: Vec<f32>,
    pub ln1_mean: Vec<f32>,
    pub ln1_var: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att_scores: Vec<f32>,
    pub att_probs: Vec<f32>,
    pub att_out: Vec<f32>,
    pub mha_res: Vec<f32>,
    pub ln2: Vec<f32>,
    pub ln2_mean: Vec<f32>,
    pub ln2_var: Vec<f32>,
    pub ff1: Vec<f32>,
    pub ff2: Vec<f32>,
    pub x_final: Vec<f32>,
}

pub fn forward(model: &Model, seq: &[u8]) -> (Vec<f32>, Cache) {
    let mut x = vec![0.0; MAX_SEQ * D_MODEL];
    for t in 0..MAX_SEQ {
        let id = seq[t] as usize;
        for d in 0..D_MODEL {
            x[t*D_MODEL + d] = model.token_emb.w[id*D_MODEL + d] + model.pos_emb.w[t*D_MODEL + d];
        }
    }

    let mut x_cur = x.clone();

    let ln1 = vec![0.0; MAX_SEQ * D_MODEL];
    let ln2 = vec![0.0; MAX_SEQ * D_MODEL];

    let mut cache = Cache {
        x0: x.clone(),
        ln1,
        ln1_mean: vec![0.0; MAX_SEQ],
        ln1_var: vec![0.0; MAX_SEQ],
        q: vec![0.0; MAX_SEQ * D_MODEL],
        k: vec![0.0; MAX_SEQ * D_MODEL],
        v: vec![0.0; MAX_SEQ * D_MODEL],
        att_scores: vec![0.0; MAX_SEQ * MAX_SEQ],
        att_probs: vec![0.0; MAX_SEQ * MAX_SEQ],
        att_out: vec![0.0; MAX_SEQ * D_MODEL],
        mha_res: vec![0.0; MAX_SEQ * D_MODEL],
        ln2,
        ln2_mean: vec![0.0; MAX_SEQ],
        ln2_var: vec![0.0; MAX_SEQ],
        ff1: vec![0.0; MAX_SEQ * D_FF],
        ff2: vec![0.0; MAX_SEQ * D_MODEL],
        x_final: vec![0.0; MAX_SEQ * D_MODEL],
    };

    for layer in 0..LAYERS {
        for t in 0..MAX_SEQ {
            let s = t * D_MODEL;
            let (m, v) = layernorm_forward(
                &x_cur[s..s + D_MODEL],
                &model.ln1_gain[layer].w,
                &model.ln1_bias[layer].w,
                &mut cache.ln1[s..s + D_MODEL],
            );
            cache.ln1_mean[t] = m;
            cache.ln1_var[t] = v;
        }

        matmul_forward(
            &cache.ln1,
            &model.q[layer].w,
            &mut cache.q,
            MAX_SEQ,
            D_MODEL,
            D_MODEL,
        );
        matmul_forward(
            &cache.ln1,
            &model.k[layer].w,
            &mut cache.k,
            MAX_SEQ,
            D_MODEL,
            D_MODEL,
        );
        matmul_forward(
            &cache.ln1,
            &model.v[layer].w,
            &mut cache.v,
            MAX_SEQ,
            D_MODEL,
            D_MODEL,
        );

        attention_scores_forward(&cache.q, &cache.k, &mut cache.att_scores);
        attention_softmax_forward(&mut cache.att_scores, &mut cache.att_probs);
        attention_apply_forward(&cache.att_probs, &cache.v, &mut cache.att_out);

        matmul_forward(
            &cache.att_out,
            &model.o[layer].w,
            &mut cache.mha_res,
            MAX_SEQ,
            D_MODEL,
            D_MODEL,
        );

        for i in 0..x_cur.len() {
            cache.mha_res[i] += x_cur[i];
        }

        x_cur = cache.mha_res.clone();

        for t in 0..MAX_SEQ {
            let s = t * D_MODEL;
            let (m, v) = layernorm_forward(
                &x_cur[s..s + D_MODEL],
                &model.ln2_gain[layer].w,
                &model.ln2_bias[layer].w,
                &mut cache.ln2[s..s + D_MODEL],
            );
            cache.ln2_mean[t] = m;
            cache.ln2_var[t] = v;
        }

        ff1_forward(&cache.ln2, &model.ffc1[layer].w, &mut cache.ff1);
        let ff1_copy = cache.ff1.clone();
        gelu_forward(&mut cache.ff1);
        ff2_forward(&cache.ff1, &model.ffc2[layer].w, &mut cache.ff2);

        for i in 0..x_cur.len() {
            cache.ff2[i] += x_cur[i];
        }

        x_cur = cache.ff2.clone();
    }

    cache.x_final = x_cur.clone();

    let mut logits = vec![0.0; MAX_SEQ * VOCAB];
    matmul_forward(
        &cache.x_final,
        &model.token_emb.w[0..(D_MODEL * VOCAB)],
        &mut logits,
        MAX_SEQ,
        D_MODEL,
        VOCAB,
    );

    (logits, cache)
}
