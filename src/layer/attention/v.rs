use crate::model::*;

pub fn v_proj(w: &LayerWeights, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; D_MODEL];
    lin(&mut out, x, &w.v.w);
    out
}

fn lin(out: &mut [f32], inp: &[f32], w: &[f32]) {
    let r = out.len();
    let c = inp.len();
    for i in 0..r {
        let mut acc = 0.0;
        for j in 0..c {
            acc += inp[j] * w[i*c + j];
        }
        out[i] = acc;
    }
}
