/**
 * Cross-Engine Robustness Dashboard (issue #7455).
 *
 * Web equivalent of src/launchers/cross_engine_dashboard.py.
 *
 * Renders engine checkboxes + perturbation config, submits a
 * POST /api/v1/analysis/cross-engine request, polls for completion,
 * then shows a Recharts BarChart of robustness scores and a metrics table.
 *
 * Config fields are validated before submission and the poll loop is
 * bounded by a deadline plus a user-facing cancel (issue #8891).
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { getApiBase } from '@/api/backend';
import type { ConfigText, PerturbationConfig } from './crossEngineConfig';
import {
  CONFIG_FIELDS,
  DEFAULT_CONFIG_TEXT,
  POLL_DEADLINE_MS,
  POLL_INTERVAL_MS,
  parseConfigText,
  validateConfigText,
} from './crossEngineConfig';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type CrossEngineTaskStatus = 'idle' | 'starting' | 'running' | 'completed' | 'failed';

export interface MetricStats {
  mean: number;
  std: number;
  cv: number;
  robustness_score: number;
}

export interface EngineMetrics {
  metrics: Record<string, MetricStats>;
}

export interface CrossEngineResult {
  engines: Record<string, EngineMetrics>;
  cv_summary: Record<string, number>;
  robustness_overall: number;
  config: {
    t_end: number;
    dt: number;
    noise_amplitude: number;
    n_trials: number;
    seed: number;
  };
}

export type { ConfigKey, ConfigText, PerturbationConfig } from './crossEngineConfig';

/** Available engine names — mirrors ENGINE_NAMES in the Python service. */
const ENGINE_NAMES = ['pendulum_stub', 'mujoco', 'drake', 'pinocchio'] as const;
type EngineName = (typeof ENGINE_NAMES)[number];

const ROBUSTNESS_COLORS: Record<string, string> = {
  pendulum_stub: '#60a5fa',
  mujoco: '#34d399',
  drake: '#f472b6',
  pinocchio: '#fbbf24',
};

const DEFAULT_COLOR = '#a78bfa';

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function startStudy(
  engines: EngineName[],
  config: PerturbationConfig,
): Promise<string> {
  const base = getApiBase();
  const resp = await fetch(`${base}/api/v1/analysis/cross-engine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ engines, config }),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to start study: ${resp.status} ${text}`);
  }
  const data = await resp.json();
  return data.task_id as string;
}

async function pollStatus(taskId: string): Promise<{ status: string; result?: CrossEngineResult; error?: string }> {
  const base = getApiBase();
  const resp = await fetch(`${base}/api/v1/analysis/cross-engine/status/${taskId}`);
  if (!resp.ok) {
    throw new Error(`Poll failed: ${resp.status}`);
  }
  return resp.json();
}

// ---------------------------------------------------------------------------
// Chart helpers
// ---------------------------------------------------------------------------

function buildRobustnessChartData(result: CrossEngineResult): { engine: string; robustness: number }[] {
  return Object.entries(result.engines).map(([engine, data]) => {
    const scores = Object.values(data.metrics).map((m) => m.robustness_score);
    const avg = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
    return { engine, robustness: parseFloat(avg.toFixed(4)) };
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CrossEngineDashboardPage() {
  const [selectedEngines, setSelectedEngines] = useState<Set<EngineName>>(
    new Set(['pendulum_stub']),
  );
  const [configText, setConfigText] = useState<ConfigText>(DEFAULT_CONFIG_TEXT);
  const [status, setStatus] = useState<CrossEngineTaskStatus>('idle');
  const [result, setResult] = useState<CrossEngineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollStartRef = useRef<number>(0);

  const fieldErrors = validateConfigText(configText);
  const hasErrors = Object.keys(fieldErrors).length > 0;
  const isBusy = status === 'running' || status === 'starting';

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const toggleEngine = useCallback((name: EngineName) => {
    setSelectedEngines((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  }, []);

  const handleRun = useCallback(async () => {
    if (selectedEngines.size === 0) return;
    // Re-validate here so a stale render can never submit a bad config.
    if (Object.keys(validateConfigText(configText)).length > 0) return;
    setStatus('starting');
    setResult(null);
    setError(null);
    setNotice(null);
    try {
      const taskId = await startStudy(
        [...selectedEngines] as EngineName[],
        parseConfigText(configText),
      );
      setStatus('running');
      pollStartRef.current = Date.now();
      pollRef.current = setInterval(async () => {
        if (Date.now() - pollStartRef.current >= POLL_DEADLINE_MS) {
          stopPolling();
          setError(
            `Gave up waiting for the study after ${Math.round(POLL_DEADLINE_MS / 60000)} ` +
              'minutes. The backend may have restarted or dropped the task — check the ' +
              'server logs, then run the comparison again.',
          );
          setStatus('failed');
          return;
        }
        try {
          const data = await pollStatus(taskId);
          if (data.status === 'completed' && data.result) {
            stopPolling();
            setResult(data.result);
            setStatus('completed');
          } else if (data.status === 'failed') {
            stopPolling();
            setError(data.error ?? 'Study failed');
            setStatus('failed');
          }
        } catch (pollErr) {
          stopPolling();
          setError(String(pollErr));
          setStatus('failed');
        }
      }, POLL_INTERVAL_MS);
    } catch (startErr) {
      setError(String(startErr));
      setStatus('failed');
    }
  }, [selectedEngines, configText, stopPolling]);

  const handleCancel = useCallback(() => {
    stopPolling();
    setStatus('idle');
    setNotice(
      'Study cancelled — stopped waiting for results. Any work already ' +
        'queued on the backend finishes on its own.',
    );
  }, [stopPolling]);

  const chartData = result ? buildRobustnessChartData(result) : [];

  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-gray-100 p-6 gap-6">
      {/* Header */}
      <div>
        <h1 className="heading-page text-gray-100">Cross-Engine Robustness Dashboard</h1>
        <p className="text-sm text-gray-400 mt-1">
          Perturbation-based comparison across physics engines. Equivalent to the PyQt6 desktop
          dashboard.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 flex flex-col gap-4">
        {/* Engine selector */}
        <div>
          <h2 className="text-sm font-semibold text-gray-200 mb-2">Engines</h2>
          <div className="flex flex-wrap gap-3">
            {ENGINE_NAMES.map((name) => (
              <label key={name} className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={selectedEngines.has(name)}
                  onChange={() => toggleEngine(name)}
                  className="accent-blue-500"
                  aria-label={`Include ${name} engine`}
                />
                <span className="text-sm text-gray-300">{name}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Config */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {CONFIG_FIELDS.map(({ key, label, min, step }) => (
            <div key={key} className="flex flex-col gap-1">
              <label htmlFor={`cfg-${key}`} className="text-xs text-gray-400">
                {label}
              </label>
              <input
                id={`cfg-${key}`}
                type="number"
                min={min}
                step={step}
                value={configText[key]}
                aria-invalid={fieldErrors[key] !== undefined}
                aria-describedby={fieldErrors[key] !== undefined ? `cfg-${key}-error` : undefined}
                onChange={(e) => {
                  const { value } = e.target;
                  setConfigText((c) => ({ ...c, [key]: value }));
                }}
                className={`bg-gray-700 border rounded px-2 py-1 text-sm text-gray-100 w-full ${
                  fieldErrors[key] !== undefined ? 'border-red-500' : 'border-gray-600'
                }`}
              />
              {fieldErrors[key] !== undefined && (
                <span id={`cfg-${key}-error`} role="alert" className="text-xs text-red-400">
                  {fieldErrors[key]}
                </span>
              )}
            </div>
          ))}
        </div>

        {/* Run / cancel buttons */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={handleRun}
            disabled={selectedEngines.size === 0 || hasErrors || isBusy}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded transition-colors"
            aria-busy={status === 'running'}
          >
            {status === 'starting'
              ? 'Starting…'
              : status === 'running'
                ? 'Running…'
                : 'Run comparison'}
          </button>
          {isBusy && (
            <button
              onClick={handleCancel}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 border border-gray-600 text-gray-100 text-sm font-medium rounded transition-colors"
            >
              Cancel study
            </button>
          )}
          {selectedEngines.size === 0 && (
            <span className="text-xs text-yellow-400">Select at least one engine.</span>
          )}
          {hasErrors && (
            <span className="text-xs text-yellow-400">
              Fix the highlighted configuration fields before running.
            </span>
          )}
        </div>
      </div>

      {/* Notice banner */}
      {notice && (
        <div
          className="bg-gray-800 border border-gray-600 rounded p-3 text-sm text-gray-300"
          role="status"
        >
          {notice}
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div
          className="bg-red-900/40 border border-red-700 rounded p-3 text-sm text-red-300"
          role="alert"
        >
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="flex flex-col gap-4">
          {/* Summary */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <p className="text-sm text-gray-400">
              Overall robustness:{' '}
              <span className="text-lg font-bold text-blue-300">
                {(result.robustness_overall * 100).toFixed(1)}%
              </span>
            </p>
          </div>

          {/* Bar chart */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h2 className="text-sm font-semibold text-gray-200 mb-3">
              Robustness scores by engine
            </h2>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <XAxis dataKey="engine" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ background: '#1f2937', border: '1px solid #374151' }}
                  labelStyle={{ color: '#d1d5db' }}
                  itemStyle={{ color: '#60a5fa' }}
                  formatter={(v) => {
                    const value = typeof v === 'number' ? v : Number(v ?? 0);
                    return [(value * 100).toFixed(1) + '%', 'Robustness'];
                  }}
                />
                <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 12 }} />
                <Bar dataKey="robustness" name="Robustness score" radius={[4, 4, 0, 0]}>
                  {chartData.map(({ engine }) => (
                    <Cell
                      key={engine}
                      fill={ROBUSTNESS_COLORS[engine] ?? DEFAULT_COLOR}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Metrics table */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 overflow-x-auto">
            <h2 className="text-sm font-semibold text-gray-200 mb-3">Per-engine metrics</h2>
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="py-1 pr-4 text-gray-400 font-medium">Engine</th>
                  <th className="py-1 pr-4 text-gray-400 font-medium">Metric</th>
                  <th className="py-1 pr-4 text-gray-400 font-medium">Mean</th>
                  <th className="py-1 pr-4 text-gray-400 font-medium">Std</th>
                  <th className="py-1 pr-4 text-gray-400 font-medium">CV</th>
                  <th className="py-1 text-gray-400 font-medium">Robustness</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.engines).flatMap(([engine, data]) =>
                  Object.entries(data.metrics).map(([metric, stats]) => (
                    <tr
                      key={`${engine}-${metric}`}
                      className="border-b border-gray-700/50 hover:bg-gray-700/30"
                    >
                      <td className="py-1 pr-4 text-gray-300 font-mono">{engine}</td>
                      <td className="py-1 pr-4 text-gray-400">{metric}</td>
                      <td className="py-1 pr-4 text-gray-300 font-mono">
                        {stats.mean.toExponential(3)}
                      </td>
                      <td className="py-1 pr-4 text-gray-300 font-mono">
                        {stats.std.toExponential(3)}
                      </td>
                      <td className="py-1 pr-4 text-gray-300 font-mono">
                        {stats.cv.toFixed(4)}
                      </td>
                      <td className="py-1 text-blue-300 font-mono">
                        {(stats.robustness_score * 100).toFixed(1)}%
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
