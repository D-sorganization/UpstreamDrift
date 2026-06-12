/**
 * CounterfactualPanel — ZTCF / ZVCF / induced-acceleration analyses
 * (issue #7450, parity with the PyQt6 dashboard's "Compute Analysis"
 * post-hoc path).
 *
 * - Kind selector via HelpfulField (metadata-driven from
 *   `counterfactual.kind` in field_metadata.yaml — single source).
 * - Capability gating is data-driven from
 *   GET /api/analysis/counterfactual/kinds (engine surface probe);
 *   unsupported kinds disable the Run button with an explanation.
 * - Run -> task progress (shared /simulate/status poller) -> Recharts
 *   line chart of per-DoF accelerations + summary stats.
 */

import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { HelpfulField } from '../ux/HelpfulField';
import {
  useCounterfactual,
  type CounterfactualResult,
} from '@/api/useCounterfactual';

const SERIES_COLORS = ['#60a5fa', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#2dd4bf'];
const MAX_SERIES = 6;

interface SummaryStats {
  frames: number;
  dofs: number;
  peakAbs: number;
  peakTime: number;
}

function computeSummary(result: CounterfactualResult): SummaryStats {
  let peakAbs = 0;
  let peakTime = result.times[0] ?? 0;
  result.values.forEach((row, i) => {
    row.forEach((v) => {
      const a = Math.abs(v);
      if (a > peakAbs) {
        peakAbs = a;
        peakTime = result.times[i] ?? 0;
      }
    });
  });
  return {
    frames: result.times.length,
    dofs: result.values[0]?.length ?? 0,
    peakAbs,
    peakTime,
  };
}

function ResultChart({ result }: { result: CounterfactualResult }) {
  const nDofs = Math.min(result.values[0]?.length ?? 0, MAX_SERIES);
  const chartData = useMemo(
    () =>
      result.times.map((t, i) => {
        const row: Record<string, number> = { time: t };
        for (let j = 0; j < nDofs; j++) {
          row[`dof_${j}`] = result.values[i]?.[j] ?? 0;
        }
        return row;
      }),
    [result, nDofs],
  );

  return (
    <div className="h-64" data-testid="counterfactual-chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="time"
            stroke="#9ca3af"
            tick={{ fontSize: 10 }}
            tickFormatter={(value: number) => `${value.toFixed(2)}s`}
          />
          <YAxis
            stroke="#9ca3af"
            tick={{ fontSize: 10 }}
            label={{
              value: result.units,
              angle: -90,
              position: 'insideLeft',
              fill: '#9ca3af',
              fontSize: 10,
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '6px',
            }}
            labelStyle={{ color: '#9ca3af' }}
            itemStyle={{ color: '#e5e7eb' }}
          />
          <Legend wrapperStyle={{ fontSize: '10px' }} />
          {Array.from({ length: nDofs }, (_, j) => (
            <Line
              key={`dof_${j}`}
              type="monotone"
              dataKey={`dof_${j}`}
              name={`DoF ${j}`}
              stroke={SERIES_COLORS[j % SERIES_COLORS.length]}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CounterfactualPanel() {
  const { support, fetchSupport, runState, result, error, run } = useCounterfactual();
  const [kind, setKind] = useState<string>('ztcf');

  useEffect(() => {
    void fetchSupport();
  }, [fetchSupport]);

  const supportedKinds = support?.kinds ?? [];
  const kindSupported = supportedKinds.includes(kind);
  const sessionAvailable = support?.session_available ?? false;
  const isBusy = runState === 'starting' || runState === 'running';
  const canRun = sessionAvailable && kindSupported && !isBusy;

  const summary = result ? computeSummary(result) : null;

  return (
    <div
      className="bg-gray-800 rounded-lg border border-gray-700 p-4"
      data-testid="counterfactual-panel"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">
          Counterfactual Analysis (ZTCF / ZVCF / Induced)
        </h3>
        {support?.engine && (
          <span className="text-xs text-gray-500">engine: {support.engine}</span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 items-end">
        <HelpfulField
          fieldId="counterfactual.kind"
          value={kind}
          onChange={setKind}
          disabled={isBusy}
        />
        <div>
          <button
            onClick={() => void run(kind)}
            disabled={!canRun}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-xs rounded transition-colors"
            data-testid="counterfactual-run"
          >
            {isBusy ? 'Computing…' : 'Run Analysis'}
          </button>
        </div>
      </div>

      {/* Capability gating messages (data-driven from the API) */}
      {support && !sessionAvailable && (
        <div className="mt-2 text-xs text-yellow-400" data-testid="no-session-note">
          No completed simulation session — run a simulation first.
        </div>
      )}
      {support && sessionAvailable && !kindSupported && (
        <div className="mt-2 text-xs text-yellow-400" data-testid="unsupported-note">
          The active engine does not support &lsquo;{kind}&rsquo;.
          {supportedKinds.length > 0
            ? ` Supported kinds: ${supportedKinds.join(', ')}.`
            : ' No counterfactual kinds are supported.'}
        </div>
      )}

      {/* Task progress */}
      {isBusy && (
        <div className="mt-3 text-xs text-blue-300" data-testid="counterfactual-progress">
          {runState === 'starting' ? 'Starting task…' : 'Task running — polling status…'}
        </div>
      )}

      {/* Errors (HTTP or task-level) */}
      {error && (
        <div className="mt-3 text-xs text-red-400" data-testid="counterfactual-error">
          {error}
        </div>
      )}

      {/* Results */}
      {result && runState === 'completed' && (
        <div className="mt-4 space-y-3" data-testid="counterfactual-result">
          <ResultChart result={result} />
          {summary && (
            <div className="grid grid-cols-4 gap-2 text-xs bg-gray-700/50 p-3 rounded">
              <div>
                <span className="text-gray-500">kind</span>
                <div className="text-gray-200 font-mono">{result.kind}</div>
              </div>
              <div>
                <span className="text-gray-500">frames × DoFs</span>
                <div className="text-gray-200 font-mono">
                  {summary.frames} × {summary.dofs}
                </div>
              </div>
              <div>
                <span className="text-gray-500">peak |accel| ({result.units})</span>
                <div className="text-gray-200 font-mono">{summary.peakAbs.toFixed(4)}</div>
              </div>
              <div>
                <span className="text-gray-500">peak time (s)</span>
                <div className="text-gray-200 font-mono">{summary.peakTime.toFixed(3)}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
