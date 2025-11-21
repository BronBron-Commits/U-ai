use crate::model::*;

pub fn load_model(path: &str) -> Model {
    Model::load(path)
}

pub fn run_forward(model: &mut Model, tokens: &[usize]) -> usize {
    // Simple baseline forward pass:
    // Just pick a random token for now.
    (tokens.last().unwrap() + 1) % VOCAB
}
