use crate::entropy::pool::snapshot;

pub fn estimate_noise() -> f32 {
    let snap = snapshot();
    if snap.is_empty() { return 0.0; }

    // simple variance-like measure
    let mut sum = 0.0;
    for b in snap.iter() {
        sum += (*b as f32 - 128.0).abs();
    }
    sum / snap.len() as f32
}
