mod model;
mod io;
mod layer;

use model::Model;
use io::load_or_init_model;
use layer::forward_full;

fn main() {
    let mut model = load_or_init_model("bytes.tmod");

    let mut ctx = vec![b'H' as usize];
    for _ in 0..20 {
        let next = forward_full(&mut model, &ctx);
        println!("> {}", next);
        ctx.push(next);
    }
}

// entropy injection module
mod entropy {
    pub mod client;
}
mod entropy_sampler;

// temporary override for sampling
fn sample_token(probabilities: &[f32]) -> usize {
    use rand::Rng;
    let mut rng = crate::entropy_sampler::get_rng();

    let mut cumulative = 0.0;
    let choice: f32 = rng.gen();

    for (i, p) in probabilities.iter().enumerate() {
        cumulative += *p;
        if choice < cumulative {
            return i;
        }
    }

    probabilities.len() - 1
}
