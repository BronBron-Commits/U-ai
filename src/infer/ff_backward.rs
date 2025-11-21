use crate::model::{MAX_SEQ, D_MODEL, D_FF};

pub fn gelu_backward(x: &[f32], grad: &mut [f32]) {
    for i in 0..x.len() {
        let v = x[i];
        let a = 0.79788456 * (v + 0.044715 * v * v * v);
        let h = a.tanh();
        let dh = 0.79788456 * (1.0 - h * h) * (1.0 + 0.134145 * v * v);
        let dgelu = 0.5 * (1.0 + h) + 0.5 * v * dh;
        grad[i] *= dgelu;
    }
}

pub fn ff2_backward(
    x: &[f32],
    w: &[f32],
    grad_out: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
) {
    for t in 0..MAX_SEQ {
        for j in 0..D_MODEL {
            let go = grad_out[t*D_MODEL + j];
            for i in 0..D_FF {
                dx[t*D_FF + i] += go * w[i*D_MODEL + j];
                dw[i*D_MODEL + j] += go * x[t*D_FF + i];
            }
        }
    }
}

pub fn ff1_backward(
    x: &[f32],
    w: &[f32],
    grad_out: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
) {
    for t in 0..MAX_SEQ {
        for j in 0..D_FF {
            let go = grad_out[t*D_FF + j];
            for i in 0..D_MODEL {
                dx[t*D_MODEL + i] += go * w[i*D_FF + j];
                dw[i*D_FF + j] += go * x[t*D_MODEL + i];
            }
        }
    }
}
