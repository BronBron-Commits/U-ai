use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::entropy::pool::POOL;

static mut RNG: Option<ChaCha20Rng> = None;

fn reseed() {
    let seed = {
        let pool = POOL.lock().unwrap();
        pool.snapshot32()
    };

    unsafe {
        RNG = Some(ChaCha20Rng::from_seed(seed));
    }
}

pub fn rng() -> &'static mut ChaCha20Rng {
    unsafe {
        if RNG.is_none() {
            reseed();
        } else {
            reseed();  
        }
        RNG.as_mut().unwrap()
    }
}

/// Sample an index based on probabilities, but with entropy jitter.
pub fn sample_token(probs: &[f32]) -> usize {
    let mut r = rng();
    let mut x: f32 = r.gen();

    for (i, p) in probs.iter().enumerate() {
        x -= *p;
        if x <= 0.0 {
            return i;
        }
    }

    probs.len() - 1
}
