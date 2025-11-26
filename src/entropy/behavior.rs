use crate::entropy::pool::POOL;

pub struct BehaviorProfile {
    pub temperature: f32,
    pub creativity: f32,
}

pub fn current_behavior() -> BehaviorProfile {
    let pool = POOL.lock().unwrap();
    let snap = pool.snapshot32();

    let t = (snap[0] as f32 / 255.0) * 0.8 + 0.7;
    let c = (snap[1] as f32 / 255.0);

    BehaviorProfile {
        temperature: t,
        creativity: c,
    }
}
