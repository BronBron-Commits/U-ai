fn main() {
    // Rebuild C++ when changed
    println!("cargo:rerun-if-changed=cpp/sentencepiece_bridge.cpp");

    // Compile our C++ wrapper
    cc::Build::new()
        .cpp(true)
        .file("cpp/sentencepiece_bridge.cpp")
        .flag("-std=c++17")
        .compile("sentencepiece_bridge");

    // Link our static bridge: libsentencepiece_bridge.a
    println!("cargo:rustc-link-lib=static=sentencepiece_bridge");

    // Link system SentencePiece
    println!("cargo:rustc-link-lib=dylib=sentencepiece");

    // Link libstdc++
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Search paths
    println!("cargo:rustc-link-search=native={}", std::env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
}
