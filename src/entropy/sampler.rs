use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha20Rng;

use crate::entropy::client::fetch_entropy_chunk;
use crate::entropy::pool::add_entropy;

/// Global RNG seeded by entropy + fallback
static mut RNG: Option<ChaCha20Rng> = None;

pub fn init_rng() {
    let url = "http://127.0.0.1:8080/entropy";

    let seed_bytes = fetch_entropy_chunk(url)
        .map(|b| {
            add_entropy(&b);
            b
        })
        .unwrap_or_else(|| {
            let local: [u8; 32] = rand::random();
            local.to_vec()
        });

    let mut seed32 = [0u8; 32];
    for (i, b) in seed_bytes.iter().take(32).enumerate() {
        seed32[i] = *b;
    }

    unsafe {
        RNG = Some(ChaCha20Rng::from_seed(seed32));
    }
}

pub fn get_rng() -> &'static mut ChaCha20Rng {
    unsafe {
        if RNG.is_none() {
            init_rng();
        }
        RNG.as_mut().unwrap()
    }
}

/// softmax helper
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max { max = v; }
    }

    let exp: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();

    exp.into_iter().map(|v| v / sum).collect()
}
