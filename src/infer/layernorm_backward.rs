use crate::model::D_MODEL;

pub fn layernorm_backward(
    x: &[f32],
    gain: &[f32],
    mean: f32,
    var: f32,
    grad: &mut [f32],
    gain_grad: &mut [f32],
    bias_grad: &mut [f32],
) {
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..D_MODEL {
        gain_grad[i] += grad[i] * ((x[i] - mean) * inv);
        bias_grad[i] += grad[i];
    }
    let mut dx = vec![0.0; D_MODEL];
    for i in 0..D_MODEL {
        dx[i] = grad[i] * gain[i] * inv;
    }
    grad.copy_from_slice(&dx);
}
