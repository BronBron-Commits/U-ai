use crate::model::{D_MODEL};

pub fn layernorm_forward(
    x: &[f32],
    gain: &[f32],
    bias: &[f32],
    out: &mut [f32],
) -> (f32, f32) {
    let mut mean = 0.0;
    for i in 0..D_MODEL {
        mean += x[i];
    }
    mean /= D_MODEL as f32;
    let mut var = 0.0;
    for i in 0..D_MODEL {
        let d = x[i] - mean;
        var += d * d;
    }
    var /= D_MODEL as f32;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..D_MODEL {
        out[i] = (x[i] - mean) * inv * gain[i] + bias[i];
    }
    (mean, var)
}
