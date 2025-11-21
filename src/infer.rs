use std::fs::File;
use std::io::Read;

pub struct Model {
    pub vocab: Vec<String>,
    pub tok_emb: Vec<Vec<f32>>,
    pub pos_emb: Vec<Vec<f32>>,
    pub w_qkv: Vec<Vec<f32>>,
    pub w_out: Vec<Vec<f32>>,
    pub w1: Vec<Vec<f32>>,
    pub w2: Vec<Vec<f32>>,
    pub d_model: usize,
    pub d_ff: usize,
    pub heads: usize,
}

fn read_f32(f: &mut File) -> f32 {
    let mut b = [0u8; 4];
    f.read_exact(&mut b).unwrap();
    f32::from_le_bytes(b)
}

fn read_mat(f: &mut File, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut m = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            m[r][c] = read_f32(f);
        }
    }
    m
}

pub fn load(path: &str) -> Model {
    let mut f = File::open(path).unwrap();

    let mut magic = [0u8; 5];
    f.read_exact(&mut magic).unwrap();
    assert!(&magic == b"TMOD2");

    let mut buf = [0u8; 4];

    f.read_exact(&mut buf).unwrap();
    let vocab = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let d_model = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let d_ff = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let _max_tokens = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let heads = u32::from_le_bytes(buf) as usize;

    let vocab_list = vec![
        "hello","world","rust","ai","cat","dog","yes","no",
        "red","blue","sun","moon","good","bad","up","down",
        "left","right","star","end","why","how","when","who",
        "what","where","big","small","hot","cold","life","code"
    ].iter().map(|s| s.to_string()).collect();

    let tok_emb = read_mat(&mut f, vocab, d_model);
    let pos_emb = read_mat(&mut f, 128, d_model);
    let w_qkv = read_mat(&mut f, d_model, 3 * d_model);
    let w_out = read_mat(&mut f, d_model, d_model);
    let w1 = read_mat(&mut f, d_model, d_ff);
    let w2 = read_mat(&mut f, d_ff, d_model);

    Model {
        vocab: vocab_list,
        tok_emb,
        pos_emb,
        w_qkv,
        w_out,
        w1,
        w2,
        d_model,
        d_ff,
        heads,
    }
}

fn layer_norm(x: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;

    let mut var = 0.0;
    for v in x.iter() {
        let d = v - mean;
        var += d * d;
    }
    var /= n;

    let inv = 1.0 / (var + 1e-5).sqrt();

    for v in x.iter_mut() {
        *v = (*v - mean) * inv;
    }
}

fn matmul(a: &[f32], m: &Vec<Vec<f32>>) -> Vec<f32> {
    let mut out = vec![0.0; m[0].len()];
    for i in 0..a.len() {
        let v = a[i];
        for j in 0..m[i].len() {
            out[j] += v * m[i][j];
        }
    }
    out
}

fn softmax(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in v.iter_mut() {
        *x /= sum;
    }
}

pub fn infer(model: &Model, tokens: &[usize]) -> usize {
    let seq_len = tokens.len();
    let d = model.d_model;
    let heads = model.heads;
    let head_dim = d / heads;

    // Build embeddings for entire sequence
    let mut xs = Vec::new();
    for t in 0..seq_len {
        let mut x = vec![0.0; d];
        for i in 0..d {
            x[i] = model.tok_emb[tokens[t]][i] + model.pos_emb[t][i];
        }
        xs.push(x);
    }

    // Pre-attention LayerNorm
    let mut xs_norm = xs.clone();
    for x in xs_norm.iter_mut() {
        layer_norm(x);
    }

    // Compute Q, K, V for all tokens
    let mut q_list = Vec::new();
    let mut k_list = Vec::new();
    let mut v_list = Vec::new();

    for x in xs_norm.iter() {
        let qkv = matmul(x, &model.w_qkv);

        let q = qkv[0*d .. 1*d].to_vec();
        let k = qkv[1*d .. 2*d].to_vec();
        let v = qkv[2*d .. 3*d].to_vec();

        q_list.push(q);
        k_list.push(k);
        v_list.push(v);
    }

    // Multi-head causal attention
    let mut att_out_final = vec![0.0; d];

    for h in 0..heads {
        let hs = h * head_dim;
        let he = hs + head_dim;

        let q_t = &q_list[seq_len - 1][hs..he];

        let mut scores = vec![0.0; seq_len];

        for t_i in 0..seq_len {
            let k_t = &k_list[t_i][hs..he];
            let mut dot = 0.0;

            for i in 0..head_dim {
                dot += q_t[i] * k_t[i];
            }

            dot /= (head_dim as f32).sqrt();

            // Causal mask: forbid attending to future positions
            if t_i > seq_len - 1 {
                dot = f32::NEG_INFINITY;
            }

            scores[t_i] = dot;
        }

        softmax(&mut scores);

        let mut head_out = vec![0.0; head_dim];

        for t_i in 0..seq_len {
            let v_t = &v_list[t_i][hs..he];
            let a = scores[t_i];
            for i in 0..head_dim {
                head_out[i] += v_t[i] * a;
            }
        }

        for i in 0..head_dim {
            att_out_final[hs + i] = head_out[i];
        }
    }

    // Attention output projection
    let proj = matmul(&att_out_final, &model.w_out);

    // Residual 1
    let mut x_res1 = vec![0.0; d];
    for i in 0..d {
        x_res1[i] = xs[seq_len - 1][i] + proj[i];
    }

    // Pre-FFN LayerNorm
    let mut x_res1_norm = x_res1.clone();
    layer_norm(&mut x_res1_norm);

    // Feed-forward
    let mut ff1 = matmul(&x_res1_norm, &model.w1);
    for v in ff1.iter_mut() {
        *v = v.tanh();
    }

    let ff2 = matmul(&ff1, &model.w2);

    // Residual 2
    let mut x_out = vec![0.0; d];
    for i in 0..d {
        x_out[i] = x_res1[i] + ff2[i];
    }

    // Output logits
    let mut logits = x_out.clone();
    softmax(&mut logits);

    logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0
}
