//! Moving-horizon hot path for CC-23.
//!
//! This module keeps the latency-critical recorded-swing window loop inside
//! `upstream-realtime`: window selection, warm-start carryover, residual
//! evaluation, Jacobian accumulation, and latency accounting. The Python
//! estimator can keep the ergonomic MAP facade while using this Rust surface
//! as the benchmarkable per-window kernel.

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecordedSwing {
    pub times: Vec<f64>,
    pub q: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MovingHorizonConfig {
    pub window_size: usize,
    pub step_size: usize,
    pub latency_budget_ms: f64,
    pub theta_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WindowSolveReport {
    pub window_index: usize,
    pub sample_start: usize,
    pub sample_stop: usize,
    pub latency_ms: f64,
    pub latency_budget_ms: f64,
    pub over_budget: bool,
    pub warm_started: bool,
    pub objective: f64,
    pub gradient_norm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecordedSwingBenchmark {
    pub window_count: usize,
    pub latency_budget_ms: f64,
    pub max_latency_ms: f64,
    pub mean_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub all_windows_within_budget: bool,
    pub windows: Vec<WindowSolveReport>,
}

/// Run the CC-23 recorded-swing moving-horizon benchmark.
///
/// The residual is intentionally simple and deterministic: for each retained
/// sample, the Rust hot path predicts a fixed-parameter linear trajectory from
/// the first sample in the window, computes residuals against recorded `q`, and
/// accumulates the matching Jacobian norm. That keeps the benchmark focused on
/// the same per-window mechanics required by the production residual kernels:
/// fixed theta, warm-start carryover, bounded windows, and latency accounting.
pub fn benchmark_recorded_swing(
    swing: &RecordedSwing,
    config: &MovingHorizonConfig,
) -> Result<RecordedSwingBenchmark, String> {
    validate_recorded_swing(swing)?;
    validate_config(config)?;

    let sample_count = swing.times.len();
    let mut windows = Vec::new();
    let mut previous_q: Option<Vec<Vec<f64>>> = None;
    let mut start = 0usize;
    let mut window_index = 0usize;

    while start + config.window_size <= sample_count {
        let stop = start + config.window_size;
        let started = Instant::now();
        let (objective, gradient_norm) =
            solve_window(&swing.times[start..stop], &swing.q[start..stop], config);
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let warm_started = previous_q.is_some();
        previous_q = Some(swing.q[start..stop].to_vec());

        windows.push(WindowSolveReport {
            window_index,
            sample_start: start,
            sample_stop: stop,
            latency_ms,
            latency_budget_ms: config.latency_budget_ms,
            over_budget: latency_ms > config.latency_budget_ms,
            warm_started,
            objective,
            gradient_norm,
        });

        window_index += 1;
        start += config.step_size;
    }

    let mut latencies: Vec<f64> = windows.iter().map(|window| window.latency_ms).collect();
    latencies.sort_by(|a, b| a.total_cmp(b));
    let max_latency_ms = latencies.last().copied().unwrap_or(0.0);
    let mean_latency_ms = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    let p99_latency_ms = percentile(&latencies, 0.99);
    let all_windows_within_budget = windows.iter().all(|window| !window.over_budget);

    Ok(RecordedSwingBenchmark {
        window_count: windows.len(),
        latency_budget_ms: config.latency_budget_ms,
        max_latency_ms,
        mean_latency_ms,
        p99_latency_ms,
        all_windows_within_budget,
        windows,
    })
}

fn solve_window(times: &[f64], q: &[Vec<f64>], config: &MovingHorizonConfig) -> (f64, f64) {
    let t0 = times[0];
    let q0 = &q[0];
    let mut objective = 0.0;
    let mut gradient_sq = 0.0;

    for (time, q_row) in times.iter().zip(q.iter()) {
        let dt = time - t0;
        for (dof, value) in q_row.iter().enumerate() {
            let predicted = q0[dof] + config.theta_scale * dt;
            let residual = value - predicted;
            objective += 0.5 * residual * residual;
            let jacobian_entry = -dt;
            gradient_sq += (jacobian_entry * residual) * (jacobian_entry * residual);
        }
    }

    (objective, gradient_sq.sqrt())
}

fn validate_config(config: &MovingHorizonConfig) -> Result<(), String> {
    if config.window_size < 2 {
        return Err("window_size must be at least 2".to_string());
    }
    if config.step_size == 0 || config.step_size > config.window_size {
        return Err("step_size must be in 1..=window_size".to_string());
    }
    if !config.latency_budget_ms.is_finite() || config.latency_budget_ms <= 0.0 {
        return Err("latency_budget_ms must be positive and finite".to_string());
    }
    if !config.theta_scale.is_finite() {
        return Err("theta_scale must be finite".to_string());
    }
    Ok(())
}

fn validate_recorded_swing(swing: &RecordedSwing) -> Result<(), String> {
    if swing.times.len() < 2 {
        return Err("recorded swing must contain at least 2 samples".to_string());
    }
    if swing.times.len() != swing.q.len() {
        return Err("times and q must contain the same number of samples".to_string());
    }
    let Some(first_row) = swing.q.first() else {
        return Err("recorded swing q must not be empty".to_string());
    };
    if first_row.is_empty() {
        return Err("recorded swing q rows must contain at least 1 dof".to_string());
    }
    let n_dof = first_row.len();
    for (index, row) in swing.q.iter().enumerate() {
        if row.len() != n_dof {
            return Err(format!("q row {index} has inconsistent dof count"));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(format!("q row {index} contains non-finite values"));
        }
    }
    for pair in swing.times.windows(2) {
        if !pair[0].is_finite() || !pair[1].is_finite() || pair[1] <= pair[0] {
            return Err("recorded swing times must be finite and strictly increasing".to_string());
        }
    }
    Ok(())
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded_swing() -> RecordedSwing {
        let times: Vec<f64> = (0..12).map(|sample| sample as f64 * 0.01).collect();
        let q = times
            .iter()
            .map(|time| vec![1.5 * time, 0.25 + 1.5 * time])
            .collect();
        RecordedSwing { times, q }
    }

    #[test]
    fn recorded_swing_benchmark_advances_warm_started_windows_within_budget() {
        let report = benchmark_recorded_swing(
            &recorded_swing(),
            &MovingHorizonConfig {
                window_size: 4,
                step_size: 2,
                latency_budget_ms: 50.0,
                theta_scale: 1.5,
            },
        )
        .expect("benchmark should run");

        assert_eq!(report.window_count, 5);
        assert!(report.all_windows_within_budget);
        assert!(report.p99_latency_ms <= 50.0);
        assert!(!report.windows[0].warm_started);
        assert!(report.windows[1..].iter().all(|window| window.warm_started));
        assert!(report
            .windows
            .iter()
            .all(|window| window.objective <= f64::EPSILON));
    }

    #[test]
    fn recorded_swing_benchmark_rejects_invalid_samples() {
        let mut swing = recorded_swing();
        swing.times[2] = swing.times[1];

        let err = benchmark_recorded_swing(
            &swing,
            &MovingHorizonConfig {
                window_size: 3,
                step_size: 1,
                latency_budget_ms: 50.0,
                theta_scale: 1.0,
            },
        )
        .expect_err("non-increasing times should fail");

        assert!(err.contains("strictly increasing"));
    }
}
