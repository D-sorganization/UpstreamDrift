/**
 * Analysis Tools Page — Biomechanical metrics, statistics summaries,
 * and export functionality.
 */

import { useState, useCallback } from 'react';
import { useAnalysisTools } from '@/api/useAnalysisTools';
import type { MetricInfo } from '@/api/useAnalysisTools';
export type {
  AnalysisLoadState,
  ExportResult,
  MetricInfo,
  StatisticsSummary,
} from '@/api/useAnalysisTools';

// Re-export API types so test files can import them from this module.
export type {
  MetricInfo,
  StatisticsSummary,
  ExportResult,
  AnalysisLoadState,
} from '@/api/useAnalysisTools';

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

  const [exportFormat, setExportFormat] = useState('csv');

  const isLoading = loadState === 'loading';

  const handleExport = useCallback(() => {
    exportAnalysis(exportFormat);
  }, [exportFormat, exportAnalysis]);

  // Group metrics by category
  const metricsByCategory = metrics.reduce<Record<string, MetricInfo[]>>((acc, m) => {
    const cat = m.category || 'General';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(m);
    return acc;
  }, {});

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Left Sidebar - Metrics List */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-gray-200">Metrics</h2>
          <p className="text-xs text-gray-500 mt-1">{metrics.length} metric{metrics.length !== 1 ? 's' : ''} available</p>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {metrics.length === 0 && !isLoading && (
            <div className="text-xs text-gray-500 italic text-center py-4">
              No metrics loaded. Click Refresh to fetch.
            </div>
          )}

          {Object.entries(metricsByCategory).map(([category, categoryMetrics]) => (
            <div key={category} className="mb-4">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">{category}</h3>
              <div className="space-y-1.5">
                {categoryMetrics.map((m: MetricInfo) => (
                  <div key={m.id} className="p-2 bg-gray-700/30 rounded border border-gray-600">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-200">{m.name}</span>
                      {m.value != null && (
                        <span className="text-xs text-blue-400 font-mono">{m.value}</span>
                      )}
                    </div>
                    <div className="text-xs text-gray-400 mt-0.5">{m.description}</div>
                    <div className="text-xs text-gray-500 mt-0.5">Unit: {m.unit}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
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
                    {statistics.metric_count} metrics summarized
                  </div>
                  <div className="space-y-2">
                    {Object.entries(statistics.summary).map(([key, stat]) => (
                      <div key={key} className="bg-gray-700/50 p-3 rounded">
                        <div className="text-xs font-medium text-gray-300 mb-1">{key}</div>
                        <div className="grid grid-cols-5 gap-2 text-xs">
                          <div>
                            <span className="text-gray-500">min</span>
                            <div className="text-gray-200 font-mono">{stat.min.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">max</span>
                            <div className="text-gray-200 font-mono">{stat.max.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">mean</span>
                            <div className="text-gray-200 font-mono">{stat.mean.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">median</span>
                            <div className="text-gray-200 font-mono">{stat.median.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">std</span>
                            <div className="text-gray-200 font-mono">{stat.std.toFixed(3)}</div>
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
                  <label className="text-xs text-gray-400 block mb-1">Format</label>
                  <select
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value)}
                    className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1.5 text-xs border border-gray-600"
                  >
                    <option value="csv">CSV</option>
                    <option value="json">JSON</option>
                    <option value="xlsx">Excel (.xlsx)</option>
                    <option value="pdf">PDF Report</option>
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
