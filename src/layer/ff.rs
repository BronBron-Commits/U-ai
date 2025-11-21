use crate::model::*;

pub fn ff_layer(_w: &LayerWeights, x: &[f32]) -> Vec<f32> {
    // Minimal feed-forward stub
    let mut o = vec![0.0; D_MODEL];
    for i in 0..D_MODEL {
        o[i] = (x[i] * 1.1 + 0.05).tanh();
    }
    o
}
