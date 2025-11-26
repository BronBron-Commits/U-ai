use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha20Rng;

use crate::entropy::client::fetch_entropy_chunk;

/// Global RNG seeded by entropy + fallback
static mut RNG: Option<ChaCha20Rng> = None;

/// Initialize RNG using entropy from remote source
pub fn init_rng() {
    let url = "http://127.0.0.1:8080/entropy"; // your entropy endpoint
    let seed_bytes = fetch_entropy_chunk(url)
        .unwrap_or_else(|| {
            // fallback: use local random seed
            let local: [u8; 32] = rand::random();
            local.to_vec()
        });

    // reduce to 32 bytes
    let mut seed32 = [0u8; 32];
    for (i, b) in seed_bytes.iter().take(32).enumerate() {
        seed32[i] = *b;
    }

    unsafe {
        RNG = Some(ChaCha20Rng::from_seed(seed32));
    }
}

/// get mutable reference to RNG
pub fn get_rng() -> &'static mut ChaCha20Rng {
    unsafe {
        if RNG.is_none() {
            init_rng();
        }
        RNG.as_mut().unwrap()
    }
}
