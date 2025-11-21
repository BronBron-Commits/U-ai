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
    let mut b = [0u8;4];
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

    let mut magic = [0u8;5];
    f.read_exact(&mut magic).unwrap();
    assert!(&magic == b"TMOD2");

    let mut buf = [0u8;4];
    f.read_exact(&mut buf).unwrap();
    let vocab = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let d_model = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let d_ff = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let max_tokens = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let heads = u32::from_le_bytes(buf) as usize;

    let vocab_list = vec![
        "hello","world","rust","ai","cat","dog","yes","no",
        "red","blue","sun","moon","good","bad","up","down",
        "left","right","star","end","why","how","when","who",
        "what","where","big","small","hot","cold","life","code"
    ].iter().map(|s| s.to_string()).collect();

    let tok_emb = read_mat(&mut f, vocab, d_model);
    let pos_emb = read_mat(&mut f, max_tokens, d_model);
    let w_qkv = read_mat(&mut f, d_model, 3*d_model);
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
    let t = tokens.len() - 1;

    let mut x = vec![0.0; model.d_model];
    for i in 0..model.d_model {
        x[i] = model.tok_emb[tokens[t]][i] + model.pos_emb[t][i];
    }

    // --------------------------
    // Pre-norm → Attention
    // --------------------------
    let mut x_norm = x.clone();
    layer_norm(&mut x_norm);

    let qkv = matmul(&x_norm, &model.w_qkv);

    let d = model.d_model;
    let heads = model.heads;
    let head_dim = d / heads;

    let mut att_out = vec![0.0; d];

    for h in 0..heads {
        let qs = h * head_dim;
        let qe = qs + head_dim;

        let ks = d + qs;
        let ke = ks + head_dim;

        let vs = 2*d + qs;
        let ve = vs + head_dim;

        let q = &qkv[qs..qe];
        let k = &qkv[ks..ke];
        let v = &qkv[vs..ve];

        let mut score = 0.0;
        for i in 0..head_dim {
            score += q[i] * k[i];
        }
        score /= (head_dim as f32).sqrt();

        let att = score.exp();

        for i in 0..head_dim {
            att_out[qs + i] = v[i] * att;
        }
    }

    let proj = matmul(&att_out, &model.w_out);

    // Residual 1
    let mut x1 = vec![0.0; d];
    for i in 0..d {
        x1[i] = x[i] + proj[i];
    }

    // --------------------------
    // Pre-norm → Feed-forward
    // --------------------------
    let mut x1_norm = x1.clone();
    layer_norm(&mut x1_norm);

    let mut ff1 = matmul(&x1_norm, &model.w1);
    for v in ff1.iter_mut() {
        *v = v.tanh();
    }

    let ff2 = matmul(&ff1, &model.w2);

    // Residual 2
    let mut x2 = vec![0.0; d];
    for i in 0..d {
        x2[i] = x1[i] + ff2[i];
    }

    // logits
    let mut logits = x2.clone();
    softmax(&mut logits);

    logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0
}
