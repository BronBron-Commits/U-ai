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
}

fn read_f32(f: &mut File) -> f32 {
    let mut b = [0u8;4];
    f.read_exact(&mut b).unwrap();
    f32::from_le_bytes(b)
}

fn read_mat(f: &mut File, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut m = vec![vec![0f32; cols]; rows];
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

    let vocab_list = vec![
        "hello","world","rust","ai","cat","dog","yes","no",
        "red","blue","sun","moon","good","bad","up","down",
        "left","right","star","end","why","how","when","who",
        "what","where","big","small","hot","cold","life","code"
    ].iter().map(|s| s.to_string()).collect();

    let tok_emb = read_mat(&mut f, vocab, d_model);
    let pos_emb = read_mat(&mut f, max_tokens, d_model);
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
    }
}

fn matmul(a: &[f32], m: &Vec<Vec<f32>>) -> Vec<f32> {
    let mut out = vec![0f32; m[0].len()];
    for (i, row) in m.iter().enumerate() {
        let v = a[i];
        for j in 0..row.len() {
            out[j] += v * row[j];
        }
    }
    out
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn infer(model: &Model, tokens: &[usize]) -> usize {
    let t = tokens.len() - 1;
    let tok_emb = &model.tok_emb[tokens[t]];
    let pos_emb = &model.pos_emb[t];

    // input = token embed + position embed
    let mut x = tok_emb.iter().zip(pos_emb).map(|(a,b)| a+b).collect::<Vec<f32>>();

    // compute Q,K,V
    let qkv = matmul(&x, &model.w_qkv);
    let d = model.d_model;

    let q = &qkv[0..d];
    let k = &qkv[d..2*d];
    let v = &qkv[2*d..3*d];

    // attention score = q·k / sqrt(d)
    let mut score = 0f32;
    for i in 0..d {
        score += q[i] * k[i];
    }
    score /= (d as f32).sqrt();

    let att = score.exp(); // single-token softmax

    // attention output = v * att
    let mut att_out = vec![0f32; d];
    for i in 0..d {
        att_out[i] = v[i] * att;
    }

    // output projection
    let mut proj = matmul(&att_out, &model.w_out);

    // FFN: xW1 → GELU → W2
    let mut ff1 = matmul(&proj, &model.w1);
    for v in ff1.iter_mut() {
        *v = v.tanh();
    }
    let ff2 = matmul(&ff1, &model.w2);

    // logits = ff2
    let mut logits = ff2.clone();
    softmax(&mut logits);

    // greedy decode
    logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0
}
