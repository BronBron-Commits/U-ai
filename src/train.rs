use std::fs;

pub struct CharDataset {
    pub chars: Vec<char>,
    pub stoi: std::collections::HashMap<char, usize>,
    pub itos: Vec<char>,
    pub data: Vec<usize>,
}

impl CharDataset {
    pub fn load(path: &str) -> Self {
        let text = fs::read_to_string(path).expect("Failed to read dataset file");

        // unique characters
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut stoi = std::collections::HashMap::new();
        for (i, ch) in chars.iter().enumerate() {
            stoi.insert(*ch, i);
        }

        let itos = chars.clone();

        let data: Vec<usize> = text
            .chars()
            .filter_map(|c| stoi.get(&c).cloned())
            .collect();

        Self { chars, stoi, itos, data }
    }

    // Create mini-batches for training
    pub fn get_batch(&self, batch_size: usize, seq_len: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();

        let data_len = self.data.len();

        for _ in 0..batch_size {
            let idx = rand::random::<usize>() % (data_len - seq_len - 1);
            let x = self.data[idx..idx + seq_len].to_vec();
            let y = self.data[idx + 1..idx + seq_len + 1].to_vec();
            xs.push(x);
            ys.push(y);
        }

        (xs, ys)
    }
}
use crate::infer::Model;

// Utility
fn matmul_vec(a: &[f32], m: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![0.0; m[0].len()];
    for i in 0..a.len() {
        let v = a[i];
        for j in 0..m[i].len() {
            out[j] += v * m[i][j];
        }
    }
    out
}

// LayerNorm used inside training too
fn layer_norm_train(x: &mut [f32], gain: &[f32], bias: &[f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;

    let mut var = 0.0;
    for v in x.iter() {
        let d = v - mean;
        var += d*d;
    }
    var /= n;

    let inv = 1.0 / (var + 1e-5).sqrt();

    for i in 0..x.len() {
        x[i] = (x[i] - mean) * inv * gain[i] + bias[i];
    }
}

// GELU used during training
fn gelu_train(x: &mut [f32]) {
    for v in x.iter_mut() {
        let x1 = *v;
        *v = 0.5 * x1 * (1.0 + (x1 * 0.79788456 * (1.0 + 0.044715 * x1 * x1)).tanh());
    }
}

// Forward pass for a batch
pub fn forward_batch(model: &Model, batch_x: &[Vec<usize>]) -> ForwardCache {
    let bsz = batch_x.len();
    let seq_len = batch_x[0].len();
    let d = model.d_model;
    let heads = model.heads;
    let head_dim = d / heads;

    let mut x_embed = vec![vec![vec![0.0; d]; seq_len]; bsz];
    let mut x_ln1  = x_embed.clone();
    let mut q      = x_embed.clone();
    let mut k      = x_embed.clone();
    let mut v      = x_embed.clone();
    let mut att_out= x_embed.clone();
    let mut x_res1 = x_embed.clone();
    let mut x_ln2  = x_embed.clone();
    let mut ff1    = x_embed.clone();
    let mut ff2    = x_embed.clone();
    let mut logits = vec![vec![vec![0.0; model.vocab.len()]; seq_len]; bsz];

    for b in 0..bsz {
        for t in 0..seq_len {
            // token + position embeddings
            for i in 0..d {
                x_embed[b][t][i] =
                    model.tok_emb[batch_x[b][t]][i] +
                    model.pos_emb[t][i];
            }

            // pre-attention layer norm
            x_ln1[b][t] = x_embed[b][t].clone();
            layer_norm_train(&mut x_ln1[b][t], &model.att_gain, &model.att_bias);

            // Q/K/V projection
            let qkv = matmul_vec(&x_ln1[b][t], &model.w_qkv);
            q[b][t] = qkv[0*d .. 1*d].to_vec();
            k[b][t] = qkv[1*d .. 2*d].to_vec();
            v[b][t] = qkv[2*d .. 3*d].to_vec();
        }

        // multi-head causal attention
        for t in 0..seq_len {
            let mut att_vec = vec![0.0; d];

            for h in 0..heads {
                let hs = h * head_dim;
                let he = hs + head_dim;

                // query for this head
                let q_t = &q[b][t][hs..he];

                // attention scores over all previous tokens (causal)
                let mut scores = vec![0.0; t + 1];

                for t_i in 0..=t {
                    let k_t = &k[b][t_i][hs..he];
                    let mut dot = 0.0;

                    for i in 0..head_dim {
                        dot += q_t[i] * k_t[i];
                    }

                    dot /= (head_dim as f32).sqrt();
                    scores[t_i] = dot;
                }

                // softmax
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    denom += *s;
                }
                for s in scores.iter_mut() {
                    *s /= denom;
                }

                // weighted sum of V
                let mut head_out = vec![0.0; head_dim];
                for t_i in 0..=t {
                    let w = scores[t_i];
                    let v_t = &v[b][t_i][hs..he];
                    for i in 0..head_dim {
                        head_out[i] += v_t[i] * w;
                    }
                }

                // write into main attention vector
                for i in 0..head_dim {
                    att_vec[hs + i] = head_out[i];
                }
            }

            att_out[b][t] = matmul_vec(&att_vec, &model.w_out);

            // residual 1
            for i in 0..d {
                x_res1[b][t][i] = x_embed[b][t][i] + att_out[b][t][i];
            }

            // LN before FFN
            x_ln2[b][t] = x_res1[b][t].clone();
            layer_norm_train(&mut x_ln2[b][t], &model.ffn_gain, &model.ffn_bias);

            // FFN
            ff1[b][t] = matmul_vec(&x_ln2[b][t], &model.w1);
            gelu_train(&mut ff1[b][t]);
            ff2[b][t] = matmul_vec(&ff1[b][t], &model.w2);

            // residual 2
            for i in 0..d {
                x_ln2[b][t][i] = x_res1[b][t][i] + ff2[b][t][i];
            }

            // output logits (softmax not applied yet)
            logits[b][t] = x_ln2[b][t].clone();
        }
    }

    ForwardCache {
        x_embed,
        x_ln1,
        q,
        k,
        v,
        att_out,
        x_res1,
        x_ln2,
        ff1,
        ff2,
        logits,
    }
}
