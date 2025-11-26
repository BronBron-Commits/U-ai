mod chat;
mod tokenizer;
mod llm_engine;
mod train;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "--chat" => {
                let mut s = chat::ChatSession::new(5000);
                s.run();
                return;
            }
            "--train" => {
                train::train();
                return;
            }
            _ => {}
        }
    }

    println!("U-ai: use --train or --chat");
}
