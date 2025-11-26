use std::fs;
use std::collections::HashMap;

#[derive(Debug)]
pub struct ModuleHit {
    pub path: String,
    pub score: u32,
}

// Extract simple lowercase keywords
fn extract_keywords(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 2)
        .collect()
}

// Score a module file based on keyword overlap
fn score_module(query_words: &[String], content: &str) -> u32 {
    let mut score = 0;
    let module_words = extract_keywords(content);

    for qw in query_words {
        if module_words.contains(qw) {
            score += 1;
        }
    }

    score
}

// Select best modules based on score
pub fn select_modules(user_msg: &str) -> Vec<String> {
    let mut hits = Vec::new();

    let keywords = extract_keywords(user_msg);
    if keywords.is_empty() {
        return vec![];
    }

    // recursively walk modules folder
    let mut stack = vec!["modules".to_string()];

    while let Some(path) = stack.pop() {
        if let Ok(entries) = fs::read_dir(&path) {
            for e in entries.flatten() {
                let p = e.path();

                if p.is_dir() {
                    stack.push(p.to_string_lossy().to_string());
                } else if p.is_file() {
                    if let Some(ext) = p.extension() {
                        if ext == "txt" {
                            if let Ok(content) = fs::read_to_string(&p) {
                                let score = score_module(&keywords, &content);
                                if score > 0 {
                                    hits.push(ModuleHit {
                                        path: p.to_string_lossy().to_string(),
                                        score,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort highest score first
    hits.sort_by(|a,b| b.score.cmp(&a.score));

    // Return top 3
    hits.into_iter().take(3).map(|h| h.path).collect()
}

// Load full text from selected modules
pub fn load_module_texts(paths: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for p in paths {
        if let Ok(t) = fs::read_to_string(p) {
            out.push(t);
        }
    }
    out
}
