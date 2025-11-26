use std::sync::{Mutex, Arc};

const POOL_SIZE: usize = 4096;

pub struct EntropyPool {
    buf: Vec<u8>,
}

impl EntropyPool {
    pub fn new() -> Self {
        Self { buf: Vec::with_capacity(POOL_SIZE) }
    }

    pub fn mix(&mut self, bytes: &[u8]) {
        for b in bytes {
            if self.buf.len() < POOL_SIZE {
                self.buf.push(*b);
            } else {
                // rolling buffer
                self.buf.remove(0);
                self.buf.push(*b);
            }
        }
    }

    pub fn snapshot32(&self) -> [u8; 32] {
        use rand::Rng;
        let mut out = [0u8; 32];
        let mut rng = rand::thread_rng();

        for i in 0..32 {
            let idx = rng.gen_range(0..self.buf.len());
            out[i] = self.buf[idx];
        }

        out
    }
}

lazy_static::lazy_static! {
    pub static ref POOL: Arc<Mutex<EntropyPool>> = Arc::new(Mutex::new(EntropyPool::new()));
}
