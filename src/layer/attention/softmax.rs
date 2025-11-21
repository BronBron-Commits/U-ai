pub fn attn_softmax(scores: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for &s in scores {
        if s > max { max = s; }
    }

    let mut sum = 0.0;
    let mut out = vec![0.0; scores.len()];

    for i in 0..scores.len() {
        let e = (scores[i] - max).exp();
        out[i] = e;
        sum += e;
    }

    for v in out.iter_mut() {
        *v /= sum;
    }

    out
}
