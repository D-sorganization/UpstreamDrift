//! Finite-difference benchmark.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p upstream-motion-matching --bench finite_diff
//! ```
//!
//! The matched Python baseline (calling
//! `PinocchioInverseDynMatchingSolver._finite_difference` directly) lives
//! in `tests/unit/motion_pipeline/test_bench_rust_finite_diff.py` (run
//! when `upstream-motion-matching` is built with `--features python`).
//!
//! The headline trajectory shape is N=1000 frames × 30 DOFs — matches
//! the issue #5218 acceptance benchmark.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use upstream_motion_matching::finite_diff_uniform;

fn make_traj(n: usize, d: usize) -> Vec<Vec<f64>> {
    // Deterministic sine-wave trajectory: each DOF has a unique
    // frequency and phase, so the finite-difference output is
    // non-trivial (no shortcut paths that compilers might exploit).
    let dt = 1.0 / 240.0;
    (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            (0..d)
                .map(|j| {
                    let f = 1.0 + 0.5 * j as f64;
                    (2.0 * std::f64::consts::PI * f * t + 0.1 * j as f64).sin()
                })
                .collect()
        })
        .collect()
}

fn bench_finite_diff_1000x30(c: &mut Criterion) {
    let q = make_traj(1000, 30);
    let dt = 1.0 / 240.0;

    c.bench_function("finite_diff_uniform_1000x30", |b| {
        b.iter(|| {
            let r = finite_diff_uniform(black_box(&q), black_box(dt)).expect("finite diff");
            black_box(r.qdot.len());
            black_box(r.qddot.len());
        });
    });
}

criterion_group!(benches, bench_finite_diff_1000x30);
criterion_main!(benches);
