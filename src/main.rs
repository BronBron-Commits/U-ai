mod model;
mod gen_model;
mod infer;

use infer::{load_model, run_forward};

fn main() {
    if !std::path::Path::new("toy_transformer.tmod").exists() {
        println!("Generating transformer model...");
        gen_model::generate();
    }

    let mut model = load_model("toy_transformer.tmod");

    let mut tokens = vec![b'h' as usize];
    for _ in 0..20 {
        let next = run_forward(&mut model, &tokens);
        println!("> {}", next);
        tokens.push(next);
    }
}
