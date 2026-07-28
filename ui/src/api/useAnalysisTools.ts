/**
 * Analysis Tools Hook — Fetches biomechanical metrics, statistical summaries,
 * and export downloads from the backend REST API.
 *
 * Response shapes mirror the real backend contracts in
 * `src/api/routes/analysis_tools.py` (issue #7448 — the previous version of
 * this hook was written against a fictional API and the page rendered
 * nothing / crashed against the real backend):
 *  - GET  /api/analysis/metrics    -> { status, metrics: Record<string, number | number[]> }
 *  - GET  /api/analysis/statistics -> AnalysisStatisticsResponse
 *  - POST /api/analysis/export     -> streamed CSV/JSON file download
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { apiFetch } from './fetch';
import { getApiBase } from './backend';

// ── Types (matching src/api/models/responses.py) ───────────────────────

/** Raw metrics snapshot from GET /api/analysis/metrics. */
export interface MetricsSnapshot {
  status: string;
  metrics: Record<string, number | number[]>;
}

/** Per-metric statistical summary (AnalysisMetricsSummary). */
export interface MetricSummary {
  metric_name: string;
  current: number;
  minimum: number;
  maximum: number;
  mean: number;
  std_dev: number;
}

/** AnalysisStatisticsResponse from GET /api/analysis/statistics. */
export interface StatisticsSummary {
  sim_time: number;
  sample_count: number;
  metrics: MetricSummary[];
  time_series: Record<string, number[]> | null;
}

/** Only the formats the backend actually implements (no xlsx/pdf). */
export type ExportFormat = 'csv' | 'json';

export const EXPORT_FORMATS: readonly ExportFormat[] = ['csv', 'json'];

export interface ExportResult {
  format: ExportFormat;
  filename: string;
  size_bytes: number;
}

export type AnalysisLoadState = 'idle' | 'loading' | 'loaded' | 'error';

// ── Hook ───────────────────────────────────────────────────────────────

export function useAnalysisTools() {
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [statistics, setStatistics] = useState<StatisticsSummary | null>(null);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [loadState, setLoadState] = useState<AnalysisLoadState>('idle');
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const fetchMetrics = useCallback(async () => {
    setLoadState('loading');
    setError(null);
    try {
      const data = await apiFetch<MetricsSnapshot>('/api/analysis/metrics');
      if (isMountedRef.current) {
        setMetrics(data);
        setLoadState('loaded');
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch metrics');
        setLoadState('error');
      }
    }
  }, []);

  const fetchStatistics = useCallback(async () => {
    setLoadState('loading');
    setError(null);
    try {
      const data = await apiFetch<StatisticsSummary>('/api/analysis/statistics');
      if (isMountedRef.current) {
        setStatistics(data);
        setLoadState('loaded');
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch statistics');
        setLoadState('error');
      }
    }
  }, []);

  /**
   * Export analysis data. The backend streams a file (CSV or JSON), so this
   * downloads the blob via an anchor element rather than parsing JSON.
   */
  const exportAnalysis = useCallback(async (format: ExportFormat) => {
    setLoadState('loading');
    setError(null);
    try {
      const response = await fetch(`${getApiBase()}/api/analysis/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          format,
          include_metrics: true,
          include_time_series: true,
        }),
      });
      if (!response.ok) {
        let detail = `Export failed: HTTP ${response.status}`;
        try {
          const body = (await response.json()) as Record<string, unknown>;
          if (typeof body.detail === 'string') detail = body.detail;
        } catch {
          // non-JSON error body — keep generic message
        }
        throw new Error(detail);
      }
      const blob = await response.blob();
      const filename = `analysis_export.${format}`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      if (isMountedRef.current) {
        setExportResult({ format, filename, size_bytes: blob.size });
        setLoadState('loaded');
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Export failed');
        setLoadState('error');
      }
    }
  }, []);

  return {
    metrics,
    statistics,
    exportResult,
    loadState,
    error,
    fetchMetrics,
    fetchStatistics,
    exportAnalysis,
  };
}
