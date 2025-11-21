use crate::model::{MAX_SEQ, D_MODEL, D_FF};

pub fn ff1_forward(x: &[f32], w: &[f32], out: &mut [f32]) {
    for t in 0..MAX_SEQ {
        for j in 0..D_FF {
            let mut s = 0.0;
            for i in 0..D_MODEL {
                s += x[t*D_MODEL + i] * w[i*D_FF + j];
            }
            out[t*D_FF + j] = s;
        }
    }
}

pub fn gelu_forward(x: &mut [f32]) {
    for v in x.iter_mut() {
        let a = 0.79788456 * (*v + 0.044715 * *v * *v * *v);
        let h = a.tanh();
        *v = 0.5 * *v * (1.0 + h);
    }
}

pub fn ff2_forward(x: &[f32], w: &[f32], out: &mut [f32]) {
    for t in 0..MAX_SEQ {
        for j in 0..D_MODEL {
            let mut s = 0.0;
            for i in 0..D_FF {
                s += x[t*D_FF + i] * w[i*D_MODEL + j];
            }
            out[t*D_MODEL + j] = s;
        }
    }
}
