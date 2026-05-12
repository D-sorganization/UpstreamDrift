//! Throughput benchmark: batched RL kernels.
//!
//! Mirrors the acceptance criteria in UD#5216 (≥20× speedup vs Python over
//! the RL inner loop). Configurable via the `BATCH_M` / `BATCH_J` /
//! `BATCH_STEPS` constants below; defaults match the issue's "1000 muscles
//! × 1000 steps" target.
//!
//! Run with::
//!
//!     cargo bench -p upstream-muscle --bench batch
//!
//! The Python baseline is invoked out-of-process by
//! `benches/compare_python_batch.py` so we can compare numbers in
//! identical conditions.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use upstream_muscle::activation::ActivationDynamics;
use upstream_muscle::batch::{
    activation_step_batch, joint_torques_batch, muscle_force_batch, step_full,
};
use upstream_muscle::model::{HillMuscleModel, MuscleParameters, MuscleState};

const BATCH_M: usize = 1000;
const BATCH_J: usize = 10;

struct Setup {
    u: Vec<f64>,
    a_in: Vec<f64>,
    act_params: Vec<ActivationDynamics>,
    models: Vec<HillMuscleModel>,
    states: Vec<MuscleState>,
    moment_arms: Vec<f64>,
}

fn make_setup() -> Setup {
    let u: Vec<f64> = (0..BATCH_M)
        .map(|i| 0.5 + 0.5 * ((i as f64) / (BATCH_M as f64)).sin())
        .collect();
    let a_in: Vec<f64> = (0..BATCH_M)
        .map(|i| 0.1 + 0.4 * ((i as f64 + 1.0) / (BATCH_M as f64)).cos().abs())
        .collect();
    let act_params: Vec<ActivationDynamics> = vec![ActivationDynamics::default(); BATCH_M];
    let models: Vec<HillMuscleModel> = (0..BATCH_M)
        .map(|i| {
            let p = MuscleParameters::new(500.0 + 5.0 * (i as f64), 0.10, 0.20, 10.0, 0.0, 0.05)
                .expect("valid params");
            HillMuscleModel::new(p, Some(0.56))
        })
        .collect();
    let states: Vec<MuscleState> = (0..BATCH_M)
        .map(|i| MuscleState {
            activation: 0.3,
            l_ce: 0.10 + 0.001 * ((i % 50) as f64 - 25.0),
            v_ce: 0.01 * ((i as f64).sin()),
            l_mt: 0.30,
        })
        .collect();
    let moment_arms: Vec<f64> = (0..BATCH_J * BATCH_M)
        .map(|k| 0.03 * ((k as f64) * 0.01).sin())
        .collect();
    Setup {
        u,
        a_in,
        act_params,
        models,
        states,
        moment_arms,
    }
}

fn bench_batch(c: &mut Criterion) {
    let setup = make_setup();
    let (u, a_in, act_p, m_p, states, r) = (
        setup.u,
        setup.a_in,
        setup.act_params,
        setup.models,
        setup.states,
        setup.moment_arms,
    );
    let mut a_out = vec![0.0_f64; BATCH_M];
    let mut f_out = vec![0.0_f64; BATCH_M];
    let mut tau_out = vec![0.0_f64; BATCH_J];

    let mut g = c.benchmark_group("batch_rl_step");
    g.throughput(Throughput::Elements(BATCH_M as u64));

    g.bench_function("activation_step_batch_M1000", |b| {
        b.iter(|| {
            activation_step_batch(
                black_box(&u),
                black_box(&a_in),
                0.001,
                black_box(&act_p),
                &mut a_out,
            )
            .unwrap();
            black_box(&a_out);
        })
    });

    g.bench_function("muscle_force_batch_M1000", |b| {
        b.iter(|| {
            muscle_force_batch(black_box(&m_p), black_box(&states), &mut f_out).unwrap();
            black_box(&f_out);
        })
    });

    g.bench_function("joint_torques_batch_J10xM1000", |b| {
        b.iter(|| {
            joint_torques_batch(black_box(&r), BATCH_J, BATCH_M, &f_out, &mut tau_out).unwrap();
            black_box(&tau_out);
        })
    });

    g.bench_function("step_full_M1000_J10", |b| {
        b.iter(|| {
            step_full(
                black_box(&u),
                black_box(&a_in),
                0.001,
                black_box(&act_p),
                black_box(&m_p),
                black_box(&states),
                black_box(&r),
                BATCH_J,
                &mut a_out,
                &mut tau_out,
            )
            .unwrap();
            black_box((&a_out, &tau_out));
        })
    });

    g.finish();
}

criterion_group!(benches, bench_batch);
criterion_main!(benches);
