use crate::model::Model;

/// Tiny forward function: chooses next token based on w + b score
pub fn run_forward(model: &Model, token: usize) -> usize {
    if token >= model.w.len() {
        return 0;
    }

    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;

    for i in 0..model.w.len() {
        let score = model.w[i] + model.b[i];
        if score > best_score {
            best_score = score;
            best = i;
        }
    }

    best
}
