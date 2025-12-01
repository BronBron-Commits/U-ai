fn main() {
    println!("cargo:rustc-link-lib=sentencepiece");
    println!("cargo:rustc-link-lib=sentencepiece_train");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
}
