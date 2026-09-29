/**
 * Tests for CrossEngineDashboard page data structures, pure helpers and
 * the config-validation / bounded-polling behaviour from issue #8891.
 *
 * The rendering tests never reach the results view, so Recharts stays
 * unmounted and they remain fast in a jsdom/vitest environment.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';

import type { CrossEngineResult, MetricStats, PerturbationConfig } from './CrossEngineDashboard';
import { CrossEngineDashboardPage } from './CrossEngineDashboard';
import {
  DEFAULT_CONFIG_TEXT,
  POLL_DEADLINE_MS,
  POLL_INTERVAL_MS,
  validateConfigText,
} from './crossEngineConfig';

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

// ---------------------------------------------------------------------------
// Config validation (issue #8891)
// ---------------------------------------------------------------------------

describe('validateConfigText', () => {
  it('accepts the defaults', () => {
    expect(validateConfigText(DEFAULT_CONFIG_TEXT)).toEqual({});
  });

  it('rejects a zero timestep instead of coercing it', () => {
    const errors = validateConfigText({ ...DEFAULT_CONFIG_TEXT, dt: '0' });
    expect(errors.dt).toBe('Timestep (s) must be at least 0.001.');
  });

  it('rejects zero trials', () => {
    const errors = validateConfigText({ ...DEFAULT_CONFIG_TEXT, n_trials: '0' });
    expect(errors.n_trials).toBe('Trials must be at least 1.');
  });

  it('rejects a cleared field rather than reading it as 0', () => {
    const errors = validateConfigText({ ...DEFAULT_CONFIG_TEXT, dt: '   ' });
    expect(errors.dt).toBe('Timestep (s) is required.');
  });

  it('rejects non-numeric text', () => {
    const errors = validateConfigText({ ...DEFAULT_CONFIG_TEXT, t_end: 'abc' });
    expect(errors.t_end).toBe('Duration (s) must be a number.');
  });

  it('rejects fractional whole-number fields', () => {
    const errors = validateConfigText({ ...DEFAULT_CONFIG_TEXT, n_trials: '2.5' });
    expect(errors.n_trials).toBe('Trials must be a whole number.');
  });

  it('rejects dt >= t_end, which the API also refuses', () => {
    const errors = validateConfigText({ ...DEFAULT_CONFIG_TEXT, dt: '2', t_end: '1' });
    expect(errors.dt).toBe('Timestep (s) must be smaller than the duration.');
  });

  it('allows zero noise amplitude and zero seed', () => {
    const errors = validateConfigText({
      ...DEFAULT_CONFIG_TEXT,
      noise_amplitude: '0',
      seed: '0',
    });
    expect(errors).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// Rendered behaviour (issue #8891)
// ---------------------------------------------------------------------------

/** Fetch double: POST hands back a task id, GET reports "running" forever. */
function installFetchMock() {
  const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
    if (init?.method === 'POST') {
      return { ok: true, json: async () => ({ task_id: 'task-1' }) } as unknown as Response;
    }
    return { ok: true, json: async () => ({ status: 'running' }) } as unknown as Response;
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

async function startRunningStudy() {
  render(<CrossEngineDashboardPage />);
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }));
  });
  expect(screen.getByRole('button', { name: 'Running…' })).toBeInTheDocument();
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('CrossEngineDashboardPage validation', () => {
  it('blocks Run and shows a message when the timestep is zero', () => {
    const fetchMock = installFetchMock();
    render(<CrossEngineDashboardPage />);

    fireEvent.change(screen.getByLabelText('Timestep (s)'), { target: { value: '0' } });

    expect(screen.getByText('Timestep (s) must be at least 0.001.')).toBeInTheDocument();
    const runButton = screen.getByRole('button', { name: 'Run comparison' });
    expect(runButton).toBeDisabled();
    fireEvent.click(runButton);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('blocks Run when trials is zero', () => {
    const fetchMock = installFetchMock();
    render(<CrossEngineDashboardPage />);

    fireEvent.change(screen.getByLabelText('Trials'), { target: { value: '0' } });

    expect(screen.getByText('Trials must be at least 1.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run comparison' })).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('re-enables Run once the field is corrected', () => {
    installFetchMock();
    render(<CrossEngineDashboardPage />);
    const dt = screen.getByLabelText('Timestep (s)');

    fireEvent.change(dt, { target: { value: '' } });
    expect(screen.getByRole('button', { name: 'Run comparison' })).toBeDisabled();

    fireEvent.change(dt, { target: { value: '0.005' } });
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.getByRole('button', { name: 'Run comparison' })).toBeEnabled();
  });
});

describe('CrossEngineDashboardPage polling', () => {
  it('stops polling at the deadline and reports an actionable failure', async () => {
    vi.useFakeTimers();
    const fetchMock = installFetchMock();

    await startRunningStudy();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_DEADLINE_MS);
    });

    expect(screen.getByRole('alert')).toHaveTextContent(/Gave up waiting for the study/);
    const callsAtDeadline = fetchMock.mock.calls.length;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5 * POLL_INTERVAL_MS);
    });
    expect(fetchMock.mock.calls.length).toBe(callsAtDeadline);
  });

  it('lets the user cancel a study that never finishes', async () => {
    vi.useFakeTimers();
    const fetchMock = installFetchMock();

    await startRunningStudy();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3 * POLL_INTERVAL_MS);
    });
    const callsBeforeCancel = fetchMock.mock.calls.length;
    expect(callsBeforeCancel).toBeGreaterThan(1);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel study' }));
    });

    expect(screen.getByRole('status')).toHaveTextContent(/Study cancelled/);
    expect(screen.getByRole('button', { name: 'Run comparison' })).toBeEnabled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10 * POLL_INTERVAL_MS);
    });
    expect(fetchMock.mock.calls.length).toBe(callsBeforeCancel);
  });
});
