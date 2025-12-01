use u_ai::llm_engine::LLmEngine;

fn main() {
    let engine = LLmEngine::new("model.bin", "model.spm");

    let output = engine.predict("hello");
    println!("Prediction: {}", output);
}
