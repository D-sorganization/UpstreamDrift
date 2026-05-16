/**
 * Analysis Tools Hook — Fetches biomechanical metrics, statistical summaries,
 * and export functionality from the backend REST API.
 */

import { useState, useCallback, useRef } from 'react';

// ── Types ──────────────────────────────────────────────────────────────

export interface MetricInfo {
  id: string;
  name: string;
  description: string;
  unit: string;
  category: string;
  value?: number;
}

export interface StatisticsSummary {
  dataset_id: string;
  metric_count: number;
  summary: Record<string, {
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
  }>;
}

export interface ExportResult {
  format: string;
  url: string;
  filename: string;
  size_bytes: number;
}

export type AnalysisLoadState = 'idle' | 'loading' | 'loaded' | 'error';

// ── Hook ───────────────────────────────────────────────────────────────

export function useAnalysisTools() {
  const [metrics, setMetrics] = useState<MetricInfo[]>([]);
  const [statistics, setStatistics] = useState<StatisticsSummary | null>(null);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [loadState, setLoadState] = useState<AnalysisLoadState>('idle');
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  const fetchMetrics = useCallback(async () => {
    setLoadState('loading');
    setError(null);
    try {
      const res = await fetch('/api/analysis/metrics');
      if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.status}`);
      const data = await res.json();
      if (isMountedRef.current) {
        setMetrics(data.metrics ?? data ?? []);
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
      const res = await fetch('/api/analysis/statistics');
      if (!res.ok) throw new Error(`Failed to fetch statistics: ${res.status}`);
      const data = await res.json();
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

  const exportAnalysis = useCallback(async (format: string, datasetId?: string) => {
    setLoadState('loading');
    setError(null);
    try {
      const res = await fetch('/api/analysis/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format, dataset_id: datasetId }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Export failed: ${res.status}`);
      }
      const data = await res.json();
      if (isMountedRef.current) {
        setExportResult(data);
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
