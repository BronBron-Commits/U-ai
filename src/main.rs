use std::io::{self, Write};
use u_ai::llm_engine::LLmEngine;

fn main() {
    let engine = LLmEngine::new("model.tmod", "model.spm");

    println!("=== Unhidra Chat ===");
    println!("Type your message and press Enter (type 'exit' to quit)\n");

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input.eq_ignore_ascii_case("exit") {
            println!("Exiting chat...");
            break;
        }

        let reply = engine.predict(input);
        println!("Unhidra: {}", reply);
    }
}
