use crate::llm_engine::Engine;
use anyhow::Result;
use std::io::{self, Write};

pub fn run_chat(engine: &mut Engine) -> Result<()> {
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut user = String::new();
        io::stdin().read_line(&mut user)?;
        let user = user.trim();

        if user == "exit" {
            break;
        }

        let prompt = format!("User: {}\nAssistant:", user);
        let reply = engine.infer(&prompt)?;
        println!("{}", reply);
    }

    Ok(())
}
