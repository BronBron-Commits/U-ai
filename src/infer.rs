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
    pub att_gain: Vec<f32>,
    pub att_bias: Vec<f32>,
    pub ffn_gain: Vec<f32>,
    pub ffn_bias: Vec<f32>,
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

fn read_vec(f: &mut File, len: usize) -> Vec<f32> {
    (0..len).map(|_| read_f32(f)).collect()
}

pub fn load(path: &str) -> Model {
    let mut f = File::open(path).unwrap();

    let mut magic = [0u8; 5];
    f.read_exact(&mut magic).unwrap();
    assert!(&magic == b"TMOD3");

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
    let w_qkv   = read_mat(&mut f, d_model, 3*d_model);
    let w_out   = read_mat(&mut f, d_model, d_model);
    let w1      = read_mat(&mut f, d_model, d_ff);
    let w2      = read_mat(&mut f, d_ff, d_model);

    let att_gain = read_vec(&mut f, d_model);
    let att_bias = read_vec(&mut f, d_model);
    let ffn_gain = read_vec(&mut f, d_model);
    let ffn_bias = read_vec(&mut f, d_model);

    Model {
        vocab: vocab_list,
        tok_emb,
        pos_emb,
        w_qkv,
        w_out,
        w1,
        w2,
        att_gain,
        att_bias,
        ffn_gain,
        ffn_bias,
        d_model,
        d_ff,
        heads,
    }
}

fn layer_norm(x: &mut [f32], gain: &[f32], bias: &[f32]) {
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

fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let x1 = *v;
        *v = 0.5 * x1 * (1.0 + (x1 * 0.79788456 * (1.0 + 0.044715 * x1 * x1)).tanh());
    }
}

fn matmul(a: &[f32], m: &[Vec<f32>]) -> Vec<f32> {
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

    // Embedding for entire sequence
    let mut xs = Vec::new();
    for t in 0..seq_len {
        let mut x = vec![0.0; d];
        for i in 0..d {
            x[i] = model.tok_emb[tokens[t]][i] + model.pos_emb[t][i];
        }
        xs.push(x);
    }

    // Pre-attention LN
    let mut xs_norm = xs.clone();
    for x in xs_norm.iter_mut() {
        layer_norm(x, &model.att_gain, &model.att_bias);
    }

    // Compute Q/K/V
    let mut qv = Vec::new();
    let mut kv = Vec::new();
    let mut vv = Vec::new();

    for x in xs_norm.iter() {
        let qkv = matmul(x, &model.w_qkv);
        let q = qkv[0*d .. 1*d].to_vec();
        let k = qkv[1*d .. 2*d].to_vec();
        let v = qkv[2*d .. 3*d].to_vec();
        qv.push(q);
        kv.push(k);
        vv.push(v);
    }

    // Multi-head causal attention for last token
    let mut att_out_final = vec![0.0; d];

    for h in 0..heads {
        let hs = h * head_dim;
        let he = hs + head_dim;

        let q_t = &qv[seq_len - 1][hs..he];
        let mut scores = vec![0.0; seq_len];

        for t_i in 0..seq_len {
            let k_t = &kv[t_i][hs..he];
            let mut dot = 0.0;
            for i in 0..head_dim {
                dot += q_t[i] * k_t[i];
            }
            dot /= (head_dim as f32).sqrt();

            // causal mask
            if t_i > seq_len - 1 {
                dot = f32::NEG_INFINITY;
            }

            scores[t_i] = dot;
        }

        softmax(&mut scores);

        let mut head_out = vec![0.0; head_dim];
        for t_i in 0..seq_len {
            let v_t = &vv[t_i][hs..he];
            let a = scores[t_i];
            for i in 0..head_dim {
                head_out[i] += v_t[i] * a;
            }
        }

        for i in 0..head_dim {
            att_out_final[hs + i] = head_out[i];
        }
    }

    // projection
    let proj = matmul(&att_out_final, &model.w_out);

    // residual 1
    let mut x_res1 = vec![0.0; d];
    for i in 0..d {
        x_res1[i] = xs[seq_len - 1][i] + proj[i];
    }

    // pre-FFN LN
    let mut x_ln = x_res1.clone();
    layer_norm(&mut x_ln, &model.ffn_gain, &model.ffn_bias);

    // FFN
    let mut ff1 = matmul(&x_ln, &model.w1);
    gelu(&mut ff1);
    let ff2 = matmul(&ff1, &model.w2);

    // residual 2
    let mut x_out = vec![0.0; d];
    for i in 0..d {
        x_out[i] = x_res1[i] + ff2[i];
    }

    // output logits → softmax → argmax
    let mut logits = x_out;
    softmax(&mut logits);

    logits
        .iter()
        .enumerate()
        .max_by(|a,b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}
