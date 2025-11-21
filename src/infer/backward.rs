use crate::model::*;
use crate::infer::{
    attention_backward::{attention_apply_backward, softmax_backward, attention_scores_backward},
    layernorm_backward::layernorm_backward,
    ff_backward::{gelu_backward, ff1_backward, ff2_backward},
    matmul_backward::matmul_backward,
};

pub fn backward(
    model: &mut Model,
    cache: &crate::infer::forward::Cache,
    dlogits: &mut [f32],
) {
    let mut dx_final = vec![0.0; MAX_SEQ * D_MODEL];
    let mut d_out_proj_w = vec![0.0; D_MODEL * VOCAB];

    matmul_backward(
        &cache.x_final,
        &model.token_emb.w[..D_MODEL*VOCAB],
        dlogits,
        &mut dx_final,
        &mut d_out_proj_w,
        MAX_SEQ,
        D_MODEL,
        VOCAB,
    );

    for i in 0..d_out_proj_w.len() {
        model.token_emb.grad[i] += d_out_proj_w[i];
    }

    let mut x_cur_grad = dx_final.clone();
    let mut x_prev_grad = vec![0.0; MAX_SEQ * D_MODEL];

    for layer in (0..LAYERS).rev() {
        let mut d_ff2 = vec![0.0; MAX_SEQ * D_MODEL];
        let mut d_ff1 = vec![0.0; MAX_SEQ * D_FF];
        let mut d_ff2_w = vec![0.0; D_FF * D_MODEL];
        let mut d_ff1_w = vec![0.0; D_MODEL * D_FF];

        for i in 0..x_cur_grad.len() {
            d_ff2[i] = x_cur_grad[i];
        }

        ff2_backward(
            &cache.ff1,
            &model.ffc2[layer].w,
            &d_ff2,
            &mut d_ff1,
            &mut d_ff2_w,
        );

        for i in 0..d_ff2_w.len() {
            model.ffc2[layer].grad[i] += d_ff2_w[i];
        }

        let mut d_ff1_gelu = d_ff1.clone();
        gelu_backward(&cache.ff1, &mut d_ff1_gelu);

        ff1_backward(
            &cache.ln2,
            &model.ffc1[layer].w,
            &d_ff1_gelu,
            &mut x_prev_grad,
            &mut d_ff1_w,
        );

        for i in 0..d_ff1_w.len() {
            model.ffc1[layer].grad[i] += d_ff1_w[i];
        }

        let mut d_ln2 = vec![0.0; MAX_SEQ * D_MODEL];
        for t in 0..MAX_SEQ {
            let s = t * D_MODEL;
            let mut g = vec![0.0; D_MODEL];
            for i in 0..D_MODEL {
                g[i] = d_ff1_gelu[t*D_FF + 0] * 0.0;
            }
            for i in 0..D_MODEL {
                g[i] += x_prev_grad[s + i];
            }
            layernorm_backward(
                &cache.mha_res[s..s+D_MODEL],
                &model.ln2_gain[layer].w,
                cache.ln2_mean[t],
                cache.ln2_var[t],
                &mut g,
                &mut model.ln2_gain[layer].grad,
                &mut model.ln2_bias[layer].grad,
            );
            for i in 0..D_MODEL {
                d_ln2[s + i] += g[i];
            }
        }

        let mut d_mha_res = d_ln2.clone();

        let mut d_att_out = vec![0.0; MAX_SEQ * D_MODEL];
        let mut d_att_out_w = vec![0.0; D_MODEL * D_MODEL];

        matmul_backward(
            &cache.att_out,
            &model.o[layer].w,
            &d_mha_res,
            &mut d_att_out,
            &mut d_att_out_w,
            MAX_SEQ,
            D_MODEL,
            D_MODEL,
        );

        for i in 0..d_att_out_w.len() {
            model.o[layer].grad[i] += d_att_out_w[i];
        }

        let mut d_probs = vec![0.0; MAX_SEQ * MAX_SEQ];
        let mut d_v = vec![0.0; MAX_SEQ * D_MODEL];

        attention_apply_backward(
            &cache.att_probs,
            &cache.v,
            &d_att_out,
            &mut d_probs,
            &mut d_v,
        );

        softmax_backward(&cache.att_probs, &mut d_probs);

        let mut d_q = vec![0.0; MAX_SEQ * D_MODEL];
        let mut d_k = vec![0.0; MAX_SEQ * D_MODEL];

        attention_scores_backward(
            &cache.q,
            &cache.k,
            &d_probs,
            &mut d_q,
            &mut d_k,
        );

        let mut d_ln1 = vec![0.0; MAX_SEQ * D_MODEL];

        let mut d_qw = vec![0.0; D_MODEL * D_MODEL];
        let mut d_kw = vec![0.0; D_MODEL * D_MODEL];
        let mut d_vw = vec![0.0; D_MODEL * D_MODEL];

        matmul_backward(
            &cache.ln1,
            &model.q[layer].w,
            &d_q,
            &mut d_ln1,
            &mut d_qw,
            MAX_SEQ,
            D_MODEL,
            D_MODEL,
        );
        for i in 0..d_qw.len() {
            model.q[layer].grad[i] += d_qw[i];
        }

        let mut d_ln1_k = vec![0.0; MAX_SEQ * D_MODEL];
        matmul_backward(
            &cache.ln1,
            &model.k[layer].w,
            &d_k,
            &mut d_ln1_k,
            &mut d_kw,
            MAX_SEQ,
            D_MODEL,
            D_MODEL,
        );
        for i in 0..d_kw.len() {
            model.k[layer].grad[i] += d_kw[i];
        }

        let mut d_ln1_v = vec![0.0; MAX_SEQ * D_MODEL];
        matmul_backward(
            &cache.ln1,
            &model.v[layer].w,
            &d_v,
            &mut d_ln1_v,
            &mut d_vw,
            MAX_SEQ,
            D_MODEL,
            D_MODEL,
        );
        for i in 0..d_vw.len() {
            model.v[layer].grad[i] += d_vw[i];
        }

        let mut d_ln1_total = d_ln1;
        for i in 0..d_ln1_total.len() {
            d_ln1_total[i] += d_ln1_k[i] + d_ln1_v[i];
        }

        let mut d_x_prev = vec![0.0; MAX_SEQ * D_MODEL];

        for t in 0..MAX_SEQ {
            let s = t * D_MODEL;
            let mut g = d_ln1_total[s..s+D_MODEL].to_vec();
            layernorm_backward(
                &cache.x0[s..s+D_MODEL],
                &model.ln1_gain[layer].w,
                cache.ln1_mean[t],
                cache.ln1_var[t],
                &mut g,
                &mut model.ln1_gain[layer].grad,
                &mut model.ln1_bias[layer].grad,
            );
            for i in 0..D_MODEL {
                d_x_prev[s+i] += g[i];
            }
        }

        x_prev_grad = d_x_prev.clone();
        x_cur_grad = x_prev_grad.clone();
    }

    let mut d_token = vec![0.0; model.token_emb.w.len()];
    let mut d_pos = vec![0.0; model.pos_emb.w.len()];

    for t in 0..MAX_SEQ {
        let id = 0usize;
        for d in 0..D_MODEL {
            d_token[id*D_MODEL + d] += x_cur_grad[t*D_MODEL + d];
            d_pos[t*D_MODEL + d] += x_cur_grad[t*D_MODEL + d];
        }
    }

    for i in 0..d_token.len() {
        model.token_emb.grad[i] += d_token[i];
    }
    for i in 0..d_pos.len() {
        model.pos_emb.grad[i] += d_pos[i];
    }
}
