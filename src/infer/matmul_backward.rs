pub fn matmul_backward(a: &[f32], b: &[f32], grad_out: &[f32], da: &mut [f32], db: &mut [f32], m: usize, n: usize, p: usize) {
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
