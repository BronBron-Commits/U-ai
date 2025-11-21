mod gen_model;
mod infer;

use infer::{load_model, infer};

fn main() {
    // generate new model if missing
    if !std::path::Path::new("toy_model.tmod").exists() {
        println!("Generating toy model...");
        gen_model::generate();
    }

    let model = load_model("toy_model.tmod");

    let mut token = 0; // "hello"
    for _ in 0..20 {
        println!("> {}", model.vocab[token]);
        token = infer(token, &model);
    }
}
