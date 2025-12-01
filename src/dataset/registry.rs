use crate::dataset::source::TrainingSource;

pub struct DatasetRegistry {
    sources: Vec<TrainingSource>,
}

impl DatasetRegistry {
    pub fn new() -> Self {
        Self { sources: vec![] }
    }

    pub fn add_source(&mut self, src: TrainingSource) {
        self.sources.push(src);
    }

    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    pub fn total_weight(&self) -> f32 {
        self.sources.iter().map(|s| s.weight).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = &TrainingSource> {
        self.sources.iter()
    }
}
