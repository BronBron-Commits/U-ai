use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingSource {
    pub name: String,
    pub weight: f32,          // influence multiplier
    pub samples: Vec<String>, // raw text lines or documents
}

impl TrainingSource {
    pub fn new(name: &str, weight: f32, samples: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            weight,
            samples,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
