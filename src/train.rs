use crate::model::*;
use crate::infer::forward;
use crate::backprop::*;
use std::fs;

pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

pub fn adam_step(p: &mut Param, cfg: &AdamConfig, t: f32) {
    for i in 0..p.w.len() {
        let g = p.grad[i];
        p.m[i] = cfg.beta1 * p.m[i] + (1.0 - cfg.beta1) * g;
        p.v[i] = cfg.beta2 * p.v[i] + (1.0 - cfg.beta2) * (g * g);

        let m_hat = p.m[i] / (1.0 - cfg.beta1.powf(t));
        let v_hat = p.v[i] / (1.0 - cfg.beta2.powf(t));

        p.w[i] -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
    }
}

pub fn load_dataset(path: &str) -> Vec<u8> {
    let data = fs::read_to_string(path)
        .expect("dataset.txt missing");
    data.bytes().map(|b| b.min(127)).collect()
}

pub fn train(model: &mut Model, dataset_path: &str) {
    let data = load_dataset(dataset_path);

    let mut tstep: usize = 1;

    let cfg = AdamConfig {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };

    loop {
        // sample a random position
        let start = rand::random::<usize>() % (data.len() - MAX_SEQ - 1);
        let seq = &data[start..start + MAX_SEQ];
        let target = &data[start + 1..start + MAX_SEQ + 1];

        model.zero_grads();

        // forward pass
        let (logits, cache) = forward(model, seq);

        // compute loss + grad
        let mut loss = 0.0;
        let mut dlogits = vec![0.0; MAX_SEQ * VOCAB];

        for t in 0..MAX_SEQ {
            let idx = target[t] as usize;
            let offset = t * VOCAB;

            let mut maxv = -1e9;
            for i in 0..VOCAB {
                maxv = maxv.max(logits[offset + i]);
            }

            let mut sum = 0.0;
            let mut probs = vec![0.0; VOCAB];
            for i in 0..VOCAB {
                probs[i] = (logits[offset + i] - maxv).exp();
                sum += probs[i];
            }
            for i in 0..VOCAB {
                probs[i] /= sum;
            }

            loss += -probs[idx].ln().max(1e-9);

            for i in 0..VOCAB {
                dlogits[offset + i] = probs[i];
            }
            dlogits[offset + idx] -= 1.0;
        }

        // backward pass
        crate::infer::backward(model, &cache, &mut dlogits);

        // apply Adam
        for p in model.all_params_mut() {
            adam_step(p, &cfg, tstep as f32);
        }

        tstep += 1;

        if tstep % 50 == 0 {
            println!("step {}: loss={}", tstep, loss / MAX_SEQ as f32);
            model.save("trained.tmod");
        }
    }
}
