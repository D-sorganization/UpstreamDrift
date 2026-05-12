//! Microbenchmark: batched Hill-curve evaluation in Rust.
//!
//! Slice 1 of UD#5216. Runs 10_000 calls of `f_l` (the active force-length
//! Gaussian) over a 10_000-element input vector and reports the per-batch
//! wall time. The comparable Python baseline is invoked out-of-process
//! by `benches/compare_python.py`; check that script's output alongside
//! `cargo bench` numbers to see the GIL-release / native-math win.
//!
//! Run with::
//!
//!     cargo bench -p upstream-muscle

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use upstream_muscle::{f_l, f_t, f_v};

const BATCH: usize = 10_000;

fn inputs_l() -> Vec<f64> {
    (0..BATCH)
        .map(|i| 0.5 + 1.0 * (i as f64) / (BATCH as f64 - 1.0))
        .collect()
}

fn inputs_v() -> Vec<f64> {
    (0..BATCH)
        .map(|i| -1.0 + 2.0 * (i as f64) / (BATCH as f64 - 1.0))
        .collect()
}

fn inputs_t() -> Vec<f64> {
    (0..BATCH)
        .map(|i| 0.9 + 0.5 * (i as f64) / (BATCH as f64 - 1.0))
        .collect()
}

fn bench_hill(c: &mut Criterion) {
    let xs_l = inputs_l();
    let xs_v = inputs_v();
    let xs_t = inputs_t();

    let mut g = c.benchmark_group("hill_batch_10k");
    g.throughput(Throughput::Elements(BATCH as u64));

    g.bench_function("f_l", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for &x in &xs_l {
                s += f_l(black_box(x));
            }
            black_box(s)
        })
    });

    g.bench_function("f_v", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for &x in &xs_v {
                s += f_v(black_box(x));
            }
            black_box(s)
        })
    });

    g.bench_function("f_t", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for &x in &xs_t {
                s += f_t(black_box(x));
            }
            black_box(s)
        })
    });

    g.finish();
}

criterion_group!(benches, bench_hill);
criterion_main!(benches);
