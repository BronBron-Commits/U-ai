use std::fs;

/// Load all text files inside ./modules/github
/// and return them as one big string.
///
/// Example: “U-ai summary”, “Pretty Notepad”, etc.
pub fn load_github_modules() -> String {
    let mut combined = String::new();

    // directory containing .txt knowledge files
    let path = "modules/github";

    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return String::new(),
    };

    for entry in entries {
        if let Ok(ent) = entry {
            let p = ent.path();
            if p.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Ok(text) = fs::read_to_string(&p) {
                    combined.push_str("\n[MODULE]\n");
                    combined.push_str(&text);
                    combined.push_str("\n");
                }
            }
        }
    }

    combined
}

/// If the user_input mentions a project name,
/// append the relevant GitHub module text to history.
pub fn inject_modules(history: &mut String, user_input: &str) {
    let input = user_input.to_lowercase();
    let mut triggered = false;

    // keywords → match to module filenames
    let modules = [
        ("u-ai", "u-ai.txt"),
        ("entropy", "u-ai.txt"),
        ("chat", "u-ai.txt"),
        ("pretty notepad", "pretty_notepad.txt"),
        ("notepad", "pretty_notepad.txt"),
        ("termux", "termux_build_dead_end.txt"),
        ("uchat", "uchat_android_client.txt"),
        ("buildtools", "uchat_buildtools.txt"),
        ("rust", "u-ai-self.txt"),
    ];

    for (keyword, file) in modules {
        if input.contains(keyword) {
            let path = format!("modules/github/{}", file);
            if let Ok(text) = fs::read_to_string(&path) {
                history.push_str("\n\n[KNOWLEDGE]\n");
                history.push_str(&text);
                history.push_str("\n");
                triggered = true;
            }
        }
    }

    // if nothing matched, do nothing
    if triggered {
        history.push_str("\n[END KNOWLEDGE]\n");
    }
}
