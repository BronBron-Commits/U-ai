pub fn gelu_backward(x: &[f32], grad: &mut [f32]) {
    for i in 0..x.len() {
        let v = x[i];
        let tanh = (0.79788456 * (v + 0.044715 * v * v * v)).tanh();
        let dtanh = 0.79788456 * (1.0 - tanh * tanh)
            * (1.0 + 0.134145 * v * v);
        let dgelu = 0.5 * (1.0 + tanh) + 0.5 * v * dtanh;
        grad[i] *= dgelu;
    }
}

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

    for i in 0..x.len() {
        gain_grad[i] += grad[i] * ((x[i] - mean) * inv);
        bias_grad[i] += grad[i];
    }

    let mut dx = vec![0.0; x.len()];
    for i in 0..x.len() {
        dx[i] = grad[i] * gain[i] * inv;
    }
    grad.copy_from_slice(&dx);
}

pub fn softmax_backward(
    probs: &[f32],
    grad: &mut [f32],
) {
    let mut gsum = 0.0;
    for i in 0..probs.len() {
        gsum += grad[i] * probs[i];
    }
    for i in 0..probs.len() {
        grad[i] = probs[i] * (grad[i] - gsum);
    }
}

pub fn matmul_backward(
    a: &[f32],
    b: &[f32],
    grad_out: &[f32],
    da: &mut [f32],
    db: &mut [f32],
    m: usize,
    n: usize,
    p: usize,
) {
    // grad_out: m x p
    // a: m x n
    // b: n x p

    for i in 0..m {
        for j in 0..p {
            let go = grad_out[i*p + j];
            for k in 0..n {
                da[i*n + k] += go * b[k*p + j];
                db[k*p + j] += go * a[i*n + k];
            }
        }
    }
}
