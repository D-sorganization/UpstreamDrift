/**
 * Static analysis plots hook (issue #7449).
 *
 * Fetches the data-driven plot-type catalogue and structured PlotData
 * from the backend's AnalysisOrchestrator endpoints. The web UI carries
 * no per-type knowledge: labels, units, and axis labels all come from
 * the payload, so plot types added to the orchestrator registry appear
 * here automatically.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { apiFetch } from './fetch';

// ── Types (mirror src/shared/python/analysis/plot_data.py) ────────────

export interface PlotSeries {
  name: string;
  x: number[];
  y: number[];
  z?: number[] | null;
  units: string;
  metadata: Record<string, unknown>;
}

export interface PlotData {
  plot_type: string;
  title: string;
  x_label: string;
  y_label: string;
  series: PlotSeries[];
  metadata: Record<string, unknown>;
}

export interface PlotTypeInfo {
  id: string;
  label: string;
}

export type PlotLoadState = 'idle' | 'loading' | 'loaded' | 'error';

// ── Hook ───────────────────────────────────────────────────────────────

export function useAnalysisPlots() {
  const [plotTypes, setPlotTypes] = useState<PlotTypeInfo[]>([]);
  const [plotData, setPlotData] = useState<PlotData | null>(null);
  const [loadState, setLoadState] = useState<PlotLoadState>('idle');
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const fetchPlotTypes = useCallback(async () => {
    setError(null);
    try {
      const data = await apiFetch<{ plot_types: PlotTypeInfo[] }>(
        '/api/analysis/plot-types',
      );
      if (isMountedRef.current) {
        setPlotTypes(data.plot_types ?? []);
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch plot types');
      }
    }
  }, []);

  const fetchPlotData = useCallback(async (plotType: string) => {
    setLoadState('loading');
    setError(null);
    try {
      const data = await apiFetch<PlotData>(
        `/api/analysis/plot-data/${encodeURIComponent(plotType)}`,
      );
      if (isMountedRef.current) {
        setPlotData(data);
        setLoadState('loaded');
      }
    } catch (err) {
      if (isMountedRef.current) {
        setPlotData(null);
        setError(err instanceof Error ? err.message : 'Failed to fetch plot data');
        setLoadState('error');
      }
    }
  }, []);

  return {
    plotTypes,
    plotData,
    loadState,
    error,
    fetchPlotTypes,
    fetchPlotData,
  };
}
