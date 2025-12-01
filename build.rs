fn main() {
    cc::Build::new()
        .file("src/tokenizer.c")
        .include("src")
        .include("include")
        .compile("spp_tokenizer");
}
