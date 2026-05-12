//! Convex hull benchmark.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p upstream-mesh --bench convex_hull
//! ```
//!
//! The matched `scipy.spatial.ConvexHull` baseline lives in
//! `tests/parity_convex_hull.py` (when compiled with `--features python`)
//! — see the PR description for the recorded speedup.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use upstream_mesh::compute_convex_hull;

fn bench_convex_hull_10k(c: &mut Criterion) {
    // Fixed seed: deterministic across machines so CI deltas are real.
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0_FFEE_5219);
    let pts: Vec<[f32; 3]> = (0..10_000)
        .map(|_| {
            [
                rng.gen_range(-1.0_f32..1.0),
                rng.gen_range(-1.0_f32..1.0),
                rng.gen_range(-1.0_f32..1.0),
            ]
        })
        .collect();

    c.bench_function("convex_hull_10k_random_unit_cube", |b| {
        b.iter(|| {
            let result = compute_convex_hull(black_box(&pts)).expect("hull");
            black_box(result.num_vertices());
        });
    });
}

criterion_group!(benches, bench_convex_hull_10k);
criterion_main!(benches);
