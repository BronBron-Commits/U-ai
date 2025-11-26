use std::sync::Mutex;

pub static POOL: Mutex<Vec<u8>> = Mutex::new(Vec::new());

pub fn add_entropy(bytes: &[u8]) {
    let mut pool = POOL.lock().unwrap();
    pool.extend_from_slice(bytes);

    // prevent unbounded memory growth
    if pool.len() > 4096 {
        pool.drain(0..2048);
    }
}

pub fn snapshot() -> Vec<u8> {
    let pool = POOL.lock().unwrap();
    pool.clone()
}
