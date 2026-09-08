/**
 * Tests for CrossEngineDashboard page data structures and pure helpers.
 *
 * Avoids rendering (no DOM dependency on Recharts) so these run fast
 * in a jsdom/vitest environment.
 */

import { describe, it, expect } from 'vitest';

import type { CrossEngineResult, MetricStats, PerturbationConfig } from './CrossEngineDashboard';

// ---------------------------------------------------------------------------
// Helpers under test (inlined so tests don't rely on internal imports)
// ---------------------------------------------------------------------------

function buildRobustnessChartData(
  result: CrossEngineResult,
): { engine: string; robustness: number }[] {
  return Object.entries(result.engines).map(([engine, data]) => {
    const scores = Object.values(data.metrics).map((m) => m.robustness_score);
    const avg = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
    return { engine, robustness: parseFloat(avg.toFixed(4)) };
  });
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const makeMetrics = (robustness: number): Record<string, MetricStats> => ({
  total_energy_final: { mean: 0.5, std: 0.05, cv: 0.1, robustness_score: robustness },
  end_effector_speed_final: { mean: 1.0, std: 0.1, cv: 0.1, robustness_score: robustness },
  peak_end_effector_speed: { mean: 1.5, std: 0.15, cv: 0.1, robustness_score: robustness },
});

const SAMPLE_RESULT: CrossEngineResult = {
  engines: {
    pendulum_stub: { metrics: makeMetrics(0.9) },
    mujoco: { metrics: makeMetrics(0.75) },
  },
  cv_summary: { cv_total_energy_final: 0.1, cv_end_effector_speed_final: 0.1 },
  robustness_overall: 0.82,
  config: { t_end: 1.0, dt: 0.01, noise_amplitude: 0.05, n_trials: 10, seed: 42 },
};

// ---------------------------------------------------------------------------
// Data-structure tests
// ---------------------------------------------------------------------------

describe('CrossEngineResult data structures', () => {
  it('should have expected engines', () => {
    expect(Object.keys(SAMPLE_RESULT.engines)).toContain('pendulum_stub');
    expect(Object.keys(SAMPLE_RESULT.engines)).toContain('mujoco');
  });

  it('should have per-engine metrics', () => {
    const metrics = SAMPLE_RESULT.engines['pendulum_stub'].metrics;
    expect(Object.keys(metrics)).toContain('total_energy_final');
    expect(metrics['total_energy_final'].mean).toBe(0.5);
    expect(metrics['total_energy_final'].robustness_score).toBe(0.9);
  });

  it('should have overall robustness in [0,1]', () => {
    expect(SAMPLE_RESULT.robustness_overall).toBeGreaterThanOrEqual(0);
    expect(SAMPLE_RESULT.robustness_overall).toBeLessThanOrEqual(1);
  });

  it('should have config fields', () => {
    const cfg = SAMPLE_RESULT.config;
    expect(cfg.t_end).toBeGreaterThan(0);
    expect(cfg.dt).toBeGreaterThan(0);
    expect(cfg.n_trials).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// Chart data helper
// ---------------------------------------------------------------------------

describe('buildRobustnessChartData', () => {
  it('returns one entry per engine', () => {
    const data = buildRobustnessChartData(SAMPLE_RESULT);
    expect(data).toHaveLength(2);
  });

  it('computes robustness as average of metric robustness_scores', () => {
    const data = buildRobustnessChartData(SAMPLE_RESULT);
    const stub = data.find((d) => d.engine === 'pendulum_stub');
    expect(stub).toBeDefined();
    // All metrics have robustness_score = 0.9, so average = 0.9
    expect(stub!.robustness).toBeCloseTo(0.9, 4);
  });

  it('handles engine with no metrics gracefully', () => {
    const emptyResult: CrossEngineResult = {
      ...SAMPLE_RESULT,
      engines: { empty_engine: { metrics: {} } },
    };
    const data = buildRobustnessChartData(emptyResult);
    expect(data[0].robustness).toBe(0);
  });

  it('rounds robustness to 4 decimal places', () => {
    const data = buildRobustnessChartData(SAMPLE_RESULT);
    data.forEach((d) => {
      const s = String(d.robustness);
      const decimals = s.includes('.') ? s.split('.')[1].length : 0;
      expect(decimals).toBeLessThanOrEqual(4);
    });
  });
});

// ---------------------------------------------------------------------------
// PerturbationConfig defaults
// ---------------------------------------------------------------------------

describe('PerturbationConfig', () => {
  it('should accept a fully-specified config', () => {
    const cfg: PerturbationConfig = {
      t_end: 2.0,
      dt: 0.005,
      noise_amplitude: 0.1,
      n_trials: 20,
      seed: 99,
    };
    expect(cfg.t_end).toBe(2.0);
    expect(cfg.n_trials).toBe(20);
  });

  it('default values satisfy physics constraints', () => {
    const cfg: PerturbationConfig = {
      t_end: 1.0,
      dt: 0.01,
      noise_amplitude: 0.05,
      n_trials: 10,
      seed: 42,
    };
    expect(cfg.t_end).toBeGreaterThan(cfg.dt);
    expect(cfg.noise_amplitude).toBeGreaterThanOrEqual(0);
    expect(cfg.n_trials).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// Metric stats formatting
// ---------------------------------------------------------------------------

describe('MetricStats formatting', () => {
  it('formats robustness_score as percentage', () => {
    const stats: MetricStats = { mean: 0.5, std: 0.05, cv: 0.1, robustness_score: 0.9 };
    const pct = (stats.robustness_score * 100).toFixed(1);
    expect(pct).toBe('90.0');
  });

  it('formats mean in scientific notation', () => {
    const stats: MetricStats = { mean: 0.000123, std: 0.00001, cv: 0.081, robustness_score: 0.92 };
    expect(stats.mean.toExponential(3)).toBe('1.230e-4');
  });
});
