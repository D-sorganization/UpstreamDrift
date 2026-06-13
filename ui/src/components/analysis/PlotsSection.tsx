/**
 * PlotsSection — post-run static analysis plots (issue #7449).
 *
 * Plot-type selector populated from `GET /api/analysis/plot-types`
 * (data-driven from the backend orchestrator registry) plus a generic
 * PlotData renderer. No per-type knowledge lives in the web app.
 */

import { useEffect, useState, useCallback } from 'react';
import { useAnalysisPlots } from '@/api/useAnalysisPlots';
import { PlotDataChart } from './PlotDataChart';

export function PlotsSection() {
  const { plotTypes, plotData, loadState, error, fetchPlotTypes, fetchPlotData } =
    useAnalysisPlots();
  const [selected, setSelected] = useState<string>('');

  useEffect(() => {
    fetchPlotTypes();
  }, [fetchPlotTypes]);

  const handleSelect = useCallback(
    (plotType: string) => {
      setSelected(plotType);
      if (plotType) {
        fetchPlotData(plotType);
      }
    },
    [fetchPlotData],
  );

  const isLoading = loadState === 'loading';

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3 gap-3">
        <h3 className="text-sm font-semibold text-gray-300">Static Plots</h3>
        <div className="flex items-center gap-2">
          <label htmlFor="plot-type-select" className="text-xs text-gray-400">
            Plot type
          </label>
          <select
            id="plot-type-select"
            value={selected}
            onChange={(e) => handleSelect(e.target.value)}
            disabled={plotTypes.length === 0}
            className="bg-gray-700 text-gray-200 rounded px-2 py-1.5 text-xs border border-gray-600"
          >
            <option value="">Select a plot…</option>
            {plotTypes.map((pt) => (
              <option key={pt.id} value={pt.id}>
                {pt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div
          className="bg-red-900/30 border border-red-800 rounded px-3 py-2 text-xs text-red-300 mb-3"
          data-testid="plots-error"
        >
          {error}
        </div>
      )}

      <div className="h-80">
        {isLoading ? (
          <div
            className="flex items-center justify-center h-full text-gray-400"
            data-testid="plots-loading"
          >
            <p className="text-sm italic">Loading plot data…</p>
          </div>
        ) : plotData ? (
          <PlotDataChart data={plotData} />
        ) : (
          !error && (
            <div className="flex items-center justify-center h-full text-gray-400">
              <p className="text-sm italic">
                Run a simulation, then pick a plot type to view post-run analysis.
              </p>
            </div>
          )
        )}
      </div>
    </div>
  );
}
