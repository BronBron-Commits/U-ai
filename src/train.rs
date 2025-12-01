use crate::model::Model;
use crate::tokenizer::Tokenizer;

pub fn train() {
    let tokenizer = Tokenizer::load("model.spm")
        .expect("Failed to load tokenizer");

    let mut model = Model::new(tokenizer.vocab_size() as usize);

    println!("Training stub complete.");
}
