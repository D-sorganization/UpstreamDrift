/**
 * AnalysisPanel - Real-time biomechanics metrics and multi-plot display.
 *
 * Displays current metrics from the simulation, statistical summaries,
 * and multiple time-series plots. Supports CSV/JSON data export.
 *
 * See issue #1203
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Download, BarChart2, Activity, TrendingUp } from 'lucide-react';

/** Analysis metric from the backend. */
interface AnalysisMetric {
  metric_name: string;
  current: number;
  minimum: number;
  maximum: number;
  mean: number;
  std_dev: number;
}

/** Statistics response from the backend. */
interface AnalysisStatistics {
  sim_time: number;
  sample_count: number;
  metrics: AnalysisMetric[];
  time_series: Record<string, number[]> | null;
}

interface Props {
  /** Whether the simulation is running */
  isRunning: boolean;
  /** Polling interval in ms (default 500) */
  pollInterval?: number;
  /** Maximum data points for time series plots */
  maxDataPoints?: number;
}

// Color palette for metric plots
const METRIC_COLORS = [
  '#60a5fa', // blue
  '#34d399', // green
  '#fbbf24', // yellow
  '#f87171', // red
  '#a78bfa', // purple
  '#2dd4bf', // teal
  '#fb923c', // orange
  '#f472b6', // pink
];

// Metrics to highlight in the dashboard
const HIGHLIGHT_METRICS = [
  'club_head_speed',
  'kinetic_energy',
  'max_velocity',
  'rms_velocity',
  'sim_time',
];

export function AnalysisPanel({
  isRunning,
  pollInterval = 500,
  maxDataPoints = 200,
}: Props) {
  const [statistics, setStatistics] = useState<AnalysisStatistics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'metrics' | 'plots' | 'export'>(
    'metrics',
  );
  const pollRef = useRef<NodeJS.Timeout | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState<
    Record<string, number>[]
  >([]);

  // Fetch metrics and statistics from the backend
  const fetchStatistics = useCallback(async () => {
    try {
      // First collect a new metric snapshot
      await fetch('/api/analysis/metrics');

      // Then get statistics
      const response = await fetch('/api/analysis/statistics');
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }
      const data: AnalysisStatistics = await response.json();
      setStatistics(data);
      setError(null);

      // Build time series chart data
      if (data.time_series) {
        const timeSeries = data.time_series;
        const maxLen = Math.max(
          ...Object.values(timeSeries).map((v) => v.length),
          0,
        );
        const chartData: Record<string, number>[] = [];
        for (let i = Math.max(0, maxLen - maxDataPoints); i < maxLen; i++) {
          const point: Record<string, number> = { index: i };
          for (const [key, values] of Object.entries(timeSeries)) {
            if (i < values.length) {
              point[key] = values[i];
            }
          }
          chartData.push(point);
        }
        setTimeSeriesData(chartData);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch');
    }
  }, [maxDataPoints]);

  // Start/stop polling when simulation runs
  useEffect(() => {
    if (isRunning) {
      fetchStatistics();
      pollRef.current = setInterval(fetchStatistics, pollInterval);
    } else {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [isRunning, pollInterval, fetchStatistics]);

  // Export handler
  const handleExport = useCallback(async (format: 'csv' | 'json') => {
    try {
      const response = await fetch('/api/analysis/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format, include_metrics: true, include_time_series: true }),
      });
      if (!response.ok) {
        throw new Error(`Export failed: ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `analysis_export.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    }
  }, []);

  // Get highlighted metrics for the dashboard
  const highlightedMetrics = useMemo(() => {
    if (!statistics) return [];
    return statistics.metrics.filter((m) =>
      HIGHLIGHT_METRICS.includes(m.metric_name),
    );
  }, [statistics]);

  // Get plottable metric keys (exclude sim_time since it is the x-axis)
  const plottableKeys = useMemo(() => {
    if (!statistics?.time_series) return [];
    return Object.keys(statistics.time_series).filter(
      (k) => k !== 'sim_time' && k !== 'index',
    );
  }, [statistics]);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      {/* Tab header */}
      <div className="flex border-b border-gray-700">
        <button
          onClick={() => setActiveTab('metrics')}
          className={`flex items-center gap-1 px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'metrics'
              ? 'bg-gray-800 text-white border-b-2 border-blue-500'
              : 'text-gray-400 hover:text-white'
          }`}
          aria-label="View metrics"
        >
          <Activity size={14} aria-hidden="true" />
          Metrics
        </button>
        <button
          onClick={() => setActiveTab('plots')}
          className={`flex items-center gap-1 px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'plots'
              ? 'bg-gray-800 text-white border-b-2 border-blue-500'
              : 'text-gray-400 hover:text-white'
          }`}
          aria-label="View plots"
        >
          <BarChart2 size={14} aria-hidden="true" />
          Plots
        </button>
        <button
          onClick={() => setActiveTab('export')}
          className={`flex items-center gap-1 px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'export'
              ? 'bg-gray-800 text-white border-b-2 border-blue-500'
              : 'text-gray-400 hover:text-white'
          }`}
          aria-label="Export data"
        >
          <Download size={14} aria-hidden="true" />
          Export
        </button>
      </div>

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-900/50 text-red-300 text-xs">
          {error}
        </div>
      )}

      {/* Tab content */}
      <div className="p-4">
        {activeTab === 'metrics' && (
          <div>
            {!statistics ? (
              <p className="text-gray-500 text-sm italic">
                {isRunning
                  ? 'Collecting metrics...'
                  : 'Start a simulation to see metrics.'}
              </p>
            ) : (
              <>
                {/* Sim time header */}
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs text-gray-400">
                    Sim Time: {statistics.sim_time.toFixed(3)}s
                  </span>
                  <span className="text-xs text-gray-400">
                    Samples: {statistics.sample_count}
                  </span>
                </div>

                {/* Highlighted metric cards */}
                <div
                  className="grid grid-cols-2 gap-2 mb-4"
                  role="group"
                  aria-label="Key metrics"
                >
                  {highlightedMetrics.map((metric) => (
                    <div
                      key={metric.metric_name}
                      className="bg-gray-800 rounded p-2"
                    >
                      <div className="text-xs text-gray-400 truncate">
                        {metric.metric_name.replace(/_/g, ' ')}
                      </div>
                      <div className="text-lg font-mono text-white">
                        {metric.current.toFixed(3)}
                      </div>
                      <div className="text-xs text-gray-500 flex justify-between">
                        <span>min: {metric.minimum.toFixed(2)}</span>
                        <span>max: {metric.maximum.toFixed(2)}</span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Full metrics table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-xs" role="table">
                    <thead>
                      <tr className="text-gray-400 border-b border-gray-700">
                        <th className="text-left py-1 px-2">Metric</th>
                        <th className="text-right py-1 px-2">Current</th>
                        <th className="text-right py-1 px-2">Mean</th>
                        <th className="text-right py-1 px-2">Std Dev</th>
                      </tr>
                    </thead>
                    <tbody>
                      {statistics.metrics.map((metric) => (
                        <tr
                          key={metric.metric_name}
                          className="text-gray-300 border-b border-gray-800"
                        >
                          <td className="py-1 px-2 font-mono">
                            {metric.metric_name}
                          </td>
                          <td className="text-right py-1 px-2 font-mono">
                            {metric.current.toFixed(4)}
                          </td>
                          <td className="text-right py-1 px-2 font-mono">
                            {metric.mean.toFixed(4)}
                          </td>
                          <td className="text-right py-1 px-2 font-mono">
                            {metric.std_dev.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'plots' && (
          <div>
            {timeSeriesData.length < 2 ? (
              <p className="text-gray-500 text-sm italic">
                {isRunning
                  ? 'Collecting data for plots...'
                  : 'Start a simulation to see time-series plots.'}
              </p>
            ) : (
              <div className="space-y-4">
                {/* Multi-series plot */}
                <div>
                  <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                    <TrendingUp
                      size={12}
                      className="inline mr-1"
                      aria-hidden="true"
                    />
                    Time Series
                  </h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={timeSeriesData}
                        margin={{
                          top: 5,
                          right: 20,
                          left: 0,
                          bottom: 5,
                        }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="#374151"
                        />
                        <XAxis
                          dataKey="index"
                          stroke="#9ca3af"
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis
                          stroke="#9ca3af"
                          tick={{ fontSize: 10 }}
                          tickFormatter={(value: number) => value.toFixed(2)}
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
                        {plottableKeys.map((key, idx) => (
                          <Line
                            key={key}
                            type="monotone"
                            dataKey={key}
                            name={key.replace(/_/g, ' ')}
                            stroke={METRIC_COLORS[idx % METRIC_COLORS.length]}
                            strokeWidth={1.5}
                            dot={false}
                            isAnimationActive={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Individual metric sparklines */}
                {plottableKeys.slice(0, 4).map((key, idx) => (
                  <div key={key}>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">
                      {key.replace(/_/g, ' ')}
                    </h4>
                    <div className="h-24">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={timeSeriesData}
                          margin={{
                            top: 2,
                            right: 10,
                            left: 0,
                            bottom: 2,
                          }}
                        >
                          <Line
                            type="monotone"
                            dataKey={key}
                            stroke={
                              METRIC_COLORS[idx % METRIC_COLORS.length]
                            }
                            strokeWidth={1.5}
                            dot={false}
                            isAnimationActive={false}
                          />
                          <YAxis
                            hide
                            domain={['auto', 'auto']}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'export' && (
          <div className="space-y-4">
            <p className="text-sm text-gray-400">
              Export simulation analysis data for offline processing.
            </p>

            <div className="flex gap-3">
              <button
                onClick={() => handleExport('csv')}
                disabled={!statistics || statistics.sample_count === 0}
                className={`flex items-center gap-2 px-4 py-2 rounded font-medium text-sm transition-colors ${
                  statistics && statistics.sample_count > 0
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                }`}
                aria-label="Export as CSV"
              >
                <Download size={16} aria-hidden="true" />
                Export CSV
              </button>
              <button
                onClick={() => handleExport('json')}
                disabled={!statistics || statistics.sample_count === 0}
                className={`flex items-center gap-2 px-4 py-2 rounded font-medium text-sm transition-colors ${
                  statistics && statistics.sample_count > 0
                    ? 'bg-green-600 hover:bg-green-700 text-white'
                    : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                }`}
                aria-label="Export as JSON"
              >
                <Download size={16} aria-hidden="true" />
                Export JSON
              </button>
            </div>

            {statistics && (
              <div className="text-xs text-gray-500">
                {statistics.sample_count} samples available ({statistics.metrics.length}{' '}
                metrics tracked)
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
