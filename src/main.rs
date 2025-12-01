use u_ai::llm_engine::LLmEngine;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).map(String::as_str).unwrap_or("hello world");

    let engine = LLmEngine::new("model.tmod", "model.spm");
    let output = engine.predict(input);

    println!("Input:  {}", input);
    println!("Reply:  {}", output);
}
