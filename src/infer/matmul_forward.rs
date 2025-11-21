pub fn matmul_forward(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, p: usize) {
    for i in 0..m {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i*n + k] * b[k*p + j];
            }
            out[i*p + j] = s;
        }
    }
}
