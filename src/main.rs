use u_ai::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).map(String::as_str).unwrap_or("hello world");

    let tokenizer = Tokenizer::load("model.spm").expect("Failed to load tokenizer");

    let ids = tokenizer.encode(input).expect("Encoding failed");
    let decoded = tokenizer.decode(&ids).expect("Decoding failed");

    println!("Input:    {}", input);
    println!("Tokens:   {:?}", ids);
    println!("Decoded:  {}", decoded);
}
