/**
 * Analysis Tools Page — Biomechanical metrics, statistics summaries,
 * and export functionality.
 *
 * Rendered exclusively from real backend data (issue #7448): metrics are the
 * live snapshot from `/api/analysis/metrics`, statistics are the computed
 * history summary from `/api/analysis/statistics`, and export downloads the
 * actual CSV/JSON file streamed by `/api/analysis/export`.
 */

import { useState, useCallback } from 'react';
import { useAnalysisTools, EXPORT_FORMATS } from '@/api/useAnalysisTools';
import type { ExportFormat } from '@/api/useAnalysisTools';
export type {
  AnalysisLoadState,
  ExportFormat,
  ExportResult,
  MetricsSnapshot,
  MetricSummary,
  StatisticsSummary,
} from '@/api/useAnalysisTools';

/** Format a metric value for display: scalars rounded, vectors summarized. */
function formatMetricValue(value: number | number[]): string {
  if (Array.isArray(value)) {
    return `[${value.length} values]`;
  }
  return value.toFixed(4);
}

/**
 * AnalysisToolsPage - Full analysis tools page.
 */
export function AnalysisToolsPage() {
  const {
    metrics,
    statistics,
    exportResult,
    loadState,
    error,
    fetchMetrics,
    fetchStatistics,
    exportAnalysis,
  } = useAnalysisTools();

  const [exportFormat, setExportFormat] = useState<ExportFormat>('csv');

  const isLoading = loadState === 'loading';

  const handleExport = useCallback(() => {
    exportAnalysis(exportFormat);
  }, [exportFormat, exportAnalysis]);

  const metricEntries = metrics ? Object.entries(metrics.metrics) : [];

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Left Sidebar - Current Metrics Snapshot */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-gray-200">Current Metrics</h2>
          <p className="text-xs text-gray-500 mt-1">
            {metricEntries.length} metric{metricEntries.length !== 1 ? 's' : ''} from live engine
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {metricEntries.length === 0 && !isLoading && (
            <div className="text-xs text-gray-500 italic text-center py-4">
              No metrics loaded. Load an engine and click Refresh to fetch a live snapshot.
            </div>
          )}

          <div className="space-y-1.5">
            {metricEntries.map(([name, value]) => (
              <div key={name} className="p-2 bg-gray-700/30 rounded border border-gray-600">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-200 font-mono">{name.replace(/_/g, ' ')}</span>
                  <span className="text-xs text-blue-400 font-mono">{formatMetricValue(value)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-4 border-t border-gray-700">
          <button
            onClick={fetchMetrics}
            disabled={isLoading}
            className="w-full py-1.5 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-gray-200 text-xs rounded transition-colors"
          >
            Refresh Metrics
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
          <h1 className="text-lg font-semibold text-gray-100">Analysis Tools</h1>
          <p className="text-xs text-gray-400 mt-1">Biomechanical metrics and statistical analysis</p>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-900/30 border-b border-red-800 px-6 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Content Area */}
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="max-w-3xl mx-auto space-y-6">
            {/* Statistics Summary */}
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-300">Statistics Summary</h3>
                <button
                  onClick={fetchStatistics}
                  disabled={isLoading}
                  className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-gray-200 text-xs rounded transition-colors"
                >
                  Load Statistics
                </button>
              </div>

              {statistics ? (
                <div data-testid="statistics-panel">
                  <div className="text-xs text-gray-400 mb-3">
                    {statistics.metrics.length} metrics summarized over {statistics.sample_count}{' '}
                    samples (sim time {statistics.sim_time.toFixed(3)}s)
                  </div>
                  {statistics.sample_count === 0 && (
                    <div className="text-xs text-gray-500 italic text-center py-2">
                      No metric history yet — run a simulation to collect samples.
                    </div>
                  )}
                  <div className="space-y-2">
                    {statistics.metrics.map((stat) => (
                      <div key={stat.metric_name} className="bg-gray-700/50 p-3 rounded">
                        <div className="text-xs font-medium text-gray-300 mb-1">{stat.metric_name}</div>
                        <div className="grid grid-cols-5 gap-2 text-xs">
                          <div>
                            <span className="text-gray-500">current</span>
                            <div className="text-gray-200 font-mono">{stat.current.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">min</span>
                            <div className="text-gray-200 font-mono">{stat.minimum.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">max</span>
                            <div className="text-gray-200 font-mono">{stat.maximum.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">mean</span>
                            <div className="text-gray-200 font-mono">{stat.mean.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">std</span>
                            <div className="text-gray-200 font-mono">{stat.std_dev.toFixed(3)}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-xs text-gray-500 italic text-center py-4">
                  Click "Load Statistics" to view summary data
                </div>
              )}
            </div>

            {/* Export Section */}
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">Export Analysis</h3>
              <div className="flex gap-2 items-end">
                <div className="flex-1">
                  <label className="text-xs text-gray-400 block mb-1" htmlFor="export-format-select">
                    Format
                  </label>
                  <select
                    id="export-format-select"
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value as ExportFormat)}
                    className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1.5 text-xs border border-gray-600"
                  >
                    {EXPORT_FORMATS.map((fmt) => (
                      <option key={fmt} value={fmt}>
                        {fmt.toUpperCase()}
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  onClick={handleExport}
                  disabled={isLoading}
                  className="px-4 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-xs rounded transition-colors"
                >
                  Export
                </button>
              </div>

              {exportResult && (
                <div className="mt-3 bg-gray-700/50 p-3 rounded text-xs" data-testid="export-result">
                  <div className="text-green-400 font-medium mb-1">Export Complete</div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><span className="text-gray-500">Format:</span> <span className="text-gray-200 font-mono">{exportResult.format}</span></div>
                    <div><span className="text-gray-500">File:</span> <span className="text-gray-200 font-mono">{exportResult.filename}</span></div>
                    <div><span className="text-gray-500">Size:</span> <span className="text-gray-200 font-mono">{(exportResult.size_bytes / 1024).toFixed(1)} KB</span></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
