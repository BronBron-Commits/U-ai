use std::fs::File;
use std::io::{Read};
use rand::Rng;

pub struct ToyModel {
    pub vocab: Vec<String>,
    pub embed: Vec<Vec<f32>>,
    pub w1: Vec<Vec<f32>>,
    pub w2: Vec<Vec<f32>>,
}

pub fn load_model(path: &str) -> ToyModel {
    let mut f = File::open(path).unwrap();

    let mut magic = [0u8;4];
    f.read_exact(&mut magic).unwrap();
    assert!(&magic == b"TMOD");

    let mut buf = [0u8;4];

    f.read_exact(&mut buf).unwrap();
    let vocab = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let embed = u32::from_le_bytes(buf) as usize;

    f.read_exact(&mut buf).unwrap();
    let hidden = u32::from_le_bytes(buf) as usize;

    let mut read_f32 = |f: &mut File| -> f32 {
        let mut b = [0u8;4];
        f.read_exact(&mut b).unwrap();
        f32::from_le_bytes(b)
    };

    // vocab
    let vocab_list = vec![
        "hello","world","rust","love",
        "cat","dog","good","bad",
        "yes","no","red","blue",
        "sun","moon","star","end"
    ].iter().map(|s| s.to_string()).collect();

    // embeddings
    let mut embed_mat = vec![vec![0f32; embed]; vocab];
    for i in 0..vocab {
        for j in 0..embed {
            embed_mat[i][j] = read_f32(&mut f);
        }
    }

    // W1
    let mut w1 = vec![vec![0f32; hidden]; embed];
    for i in 0..embed {
        for j in 0..hidden {
            w1[i][j] = read_f32(&mut f);
        }
    }

    // W2
    let mut w2 = vec![vec![0f32; vocab]; hidden];
    for i in 0..hidden {
        for j in 0..vocab {
            w2[i][j] = read_f32(&mut f);
        }
    }

    ToyModel {
        vocab: vocab_list,
        embed: embed_mat,
        w1,
        w2,
    }
}

pub fn infer(token: usize, model: &ToyModel) -> usize {
    let embed = &model.embed[token];

    // hidden = embed * W1
    let mut hidden = vec![0f32; model.w1[0].len()];
    for h in 0..hidden.len() {
        for e in 0..embed.len() {
            hidden[h] += embed[e] * model.w1[e][h];
        }
        hidden[h] = hidden[h].tanh();
    }

    // logits = hidden * W2
    let mut logits = vec![0f32; model.vocab.len()];
    for v in 0..logits.len() {
        for h in 0..hidden.len() {
            logits[v] += hidden[h] * model.w2[h][v];
        }
    }

    // greedy decode
    logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0
}
