use crate::model::Model;
use std::fs::File;
use std::io::{Write, Read};

impl Model {
    pub fn save(&self, path: &str) {
        let bytes = bincode::serialize(self)
            .expect("serialize failed");

        let mut file = File::create(path)
            .expect("failed to create model file");
        file.write_all(&bytes)
            .expect("failed to write model file");
    }

    pub fn load(path: &str) -> Self {
        let mut file = File::open(path)
            .expect("model file missing");

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .expect("could not read model file");

        bincode::deserialize(&bytes)
            .expect("invalid model format")
    }
}
