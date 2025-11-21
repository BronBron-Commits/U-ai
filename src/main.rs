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
