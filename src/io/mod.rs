use std::fs::{File};
use std::io::{Read, Write};

use crate::model::*;

pub fn load_or_init_model(path: &str) -> Model {
    if let Ok(mut f) = File::open(path) {
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        parse_model(&buf)
    } else {
        let m = init_model();
        save_model(path, &m);
        m
    }
}

fn init_model() -> Model {
    let token_emb = Param::new(VOCAB * D_MODEL);
    let pos_emb   = Param::new(MAX_SEQ * D_MODEL);

    let mut layers = Vec::new();
    for _ in 0..LAYERS {
        layers.push(LayerWeights {
            q:   Param::new(D_MODEL * D_MODEL),
            k:   Param::new(D_MODEL * D_MODEL),
            v:   Param::new(D_MODEL * D_MODEL),
            o:   Param::new(D_MODEL * D_MODEL),
            ff1: Param::new(D_MODEL * D_FF),
            ff2: Param::new(D_FF * D_MODEL),
        });
    }

    let final_proj = Param::new(D_MODEL * VOCAB);

    Model { token_emb, pos_emb, layers, final_proj }
}

fn save_model(path: &str, m: &Model) {
    let mut f = File::create(path).unwrap();

    write_param(&mut f, &m.token_emb);
    write_param(&mut f, &m.pos_emb);

    for l in &m.layers {
        write_param(&mut f, &l.q);
        write_param(&mut f, &l.k);
        write_param(&mut f, &l.v);
        write_param(&mut f, &l.o);
        write_param(&mut f, &l.ff1);
        write_param(&mut f, &l.ff2);
    }

    write_param(&mut f, &m.final_proj);
}

fn write_param(f: &mut File, p: &Param) {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            p.w.as_ptr() as *const u8,
            p.w.len() * std::mem::size_of::<f32>(),
        )
    };
    f.write_all(bytes).unwrap();
}

fn parse_model(bytes: &[u8]) -> Model {
    let mut i = 0usize;

    let read_param = |sz: usize, i: &mut usize, bytes: &[u8]| -> Param {
        let end = *i + sz * 4;
        let slice = &bytes[*i..end];
        *i = end;

        let mut w = vec![0.0; sz];
        for j in 0..sz {
            let raw = &slice[j*4..j*4+4];
            w[j] = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        }

        Param { w, grad: vec![0.0; sz], m: vec![0.0; sz], v: vec![0.0; sz] }
    };

    let token_emb = read_param(VOCAB * D_MODEL, &mut i, bytes);
    let pos_emb   = read_param(MAX_SEQ * D_MODEL, &mut i, bytes);

    let mut layers = Vec::new();
    for _ in 0..LAYERS {
        layers.push(LayerWeights {
            q:   read_param(D_MODEL * D_MODEL, &mut i, bytes),
            k:   read_param(D_MODEL * D_MODEL, &mut i, bytes),
            v:   read_param(D_MODEL * D_MODEL, &mut i, bytes),
            o:   read_param(D_MODEL * D_MODEL, &mut i, bytes),
            ff1: read_param(D_MODEL * D_FF,     &mut i, bytes),
            ff2: read_param(D_FF * D_MODEL,     &mut i, bytes),
        });
    }

    let final_proj = read_param(D_MODEL * VOCAB, &mut i, bytes);

    Model { token_emb, pos_emb, layers, final_proj }
}
