mod gen_model;
mod infer;

use infer::{load, infer};

fn main() {
    if !std::path::Path::new("toy_transformer.tmod").exists() {
        println!("Generating transformer model...");
        gen_model::generate();
    }

    let model = load("toy_transformer.tmod");

    let mut tokens = vec![0]; // "hello"
    for _ in 0..20 {
        let next = infer(&model, &tokens);
        println!("> {}", model.vocab[next]);
        tokens.push(next);
    }
}
