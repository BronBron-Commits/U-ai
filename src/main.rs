use std::io::{self, Write};
use u_ai::llm_engine::LLmEngine;

fn main() {
    let engine = LLmEngine::new("u-ai-trained.tmod", "model.spm", "vocab.txt");

    println!("=== Unhidra Chat ===");
    println!("Type your message and press Enter (type 'exit' to quit)\n");

    // Conversation history buffer
    let mut history = String::new();

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("Input error, try again.");
            continue;
        }

        let input = input.trim();
        if input.eq_ignore_ascii_case("exit") {
            println!("Exiting chat...");
            break;
        }

        // Append user input to running conversation memory
        history.push_str(&format!("You: {}\n", input));

        // Feed full history to model for context
        let prompt = format!("{}\nUnhidra:", history);
        let response = engine.predict(&prompt);

        println!("Unhidra: {}\n", response);

        // Save Unhidra’s reply in the conversation memory
        history.push_str(&format!("Unhidra: {}\n", response));

        // Prevent the buffer from growing too large
        if history.len() > 4000 {
            let tail = &history[history.len() - 4000..];
            history = tail.to_string();
        }
    }
}
