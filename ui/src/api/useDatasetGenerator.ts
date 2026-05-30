/**
 * Dataset Generator Hook — Fetches dataset generation controls, features,
 * plot types, and export formats from the backend REST API.
 *
 * Provides methods for generating datasets, importing swing data, and
 * managing dataset control parameters.
 *
 * Issue #6642: F1 (export wired), F7 (catalog loading state), F8 (apiFetch)
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { apiFetch } from './fetch';

// ── Types ──────────────────────────────────────────────────────────────

export interface FeatureInfo {
  id: string;
  name: string;
  description: string;
  category: string;
}

export interface PlotType {
  id: string;
  name: string;
  description: string;
  axes: string[];
}

export interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  mime_type: string;
}

export interface DatasetControl {
  id: string;
  name: string;
  type: string;
  value: unknown;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
}

export interface GenerateResult {
  dataset_id: string;
  name: string;
  rows: number;
  columns: string[];
  created_at: string;
}

export type DatasetLoadState = 'idle' | 'loading' | 'loaded' | 'error';

// ── Helpers ──────────────────────────────────────────────────────────────

/**
 * Normalize a catalog response into a typed array.
 *
 * The dataset catalog endpoints in `src/api/routes/dataset.py` return BARE
 * arrays. This tolerates that shape, a future `{<key>: [...]}` wrapper (for
 * any of `keys`), and any malformed/null payload (→ `[]`), so the sidebar is
 * populated against the existing API contract (review-feedback #6703).
 *
 * @param data - Parsed JSON body (array, wrapper object, or anything)
 * @param keys - Candidate wrapper keys to probe, in priority order
 * @returns A `T[]` — never throws, never returns a non-array
 */
function asCatalogArray<T>(data: unknown, keys: readonly string[]): T[] {
  if (Array.isArray(data)) return data as T[];
  if (data && typeof data === 'object') {
    for (const key of keys) {
      const value = (data as Record<string, unknown>)[key];
      if (Array.isArray(value)) return value as T[];
    }
  }
  return [];
}

// ── Hook ───────────────────────────────────────────────────────────────

export function useDatasetGenerator() {
  const [features, setFeatures] = useState<FeatureInfo[]>([]);
  const [plotTypes, setPlotTypes] = useState<PlotType[]>([]);
  const [exportFormats, setExportFormats] = useState<ExportFormat[]>([]);
  const [controls, setControls] = useState<DatasetControl[]>([]);
  const [generateResult, setGenerateResult] = useState<GenerateResult | null>(null);
  const [loadState, setLoadState] = useState<DatasetLoadState>('idle');
  // F7: separate state for the initial catalog fetch (4 concurrent requests)
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  const fetchFeatures = useCallback(async () => {
    try {
      const data = await apiFetch<unknown>('/api/dataset/features');
      const list = asCatalogArray<FeatureInfo>(data, ['features']);
      if (isMountedRef.current) setFeatures(list);
    } catch (err) {
      if (isMountedRef.current)
        setError(err instanceof Error ? err.message : 'Failed to fetch features');
    }
  }, []);

  const fetchPlotTypes = useCallback(async () => {
    try {
      const data = await apiFetch<unknown>('/api/dataset/plots/types');
      const list = asCatalogArray<PlotType>(data, ['plot_types', 'types']);
      if (isMountedRef.current) setPlotTypes(list);
    } catch (err) {
      if (isMountedRef.current)
        setError(err instanceof Error ? err.message : 'Failed to fetch plot types');
    }
  }, []);

  const fetchExportFormats = useCallback(async () => {
    try {
      const data = await apiFetch<unknown>('/api/dataset/export/formats');
      const list = asCatalogArray<ExportFormat>(data, ['formats']);
      if (isMountedRef.current) setExportFormats(list);
    } catch (err) {
      if (isMountedRef.current)
        setError(err instanceof Error ? err.message : 'Failed to fetch export formats');
    }
  }, []);

  const fetchControls = useCallback(async () => {
    try {
      const data = await apiFetch<{ controls?: DatasetControl[] }>('/api/dataset/control');
      const list = Array.isArray(data.controls) ? data.controls : [];
      if (isMountedRef.current) setControls(list);
    } catch (err) {
      if (isMountedRef.current)
        setError(err instanceof Error ? err.message : 'Failed to fetch controls');
    }
  }, []);

  const generateDataset = useCallback(async (params: Record<string, unknown>) => {
    setLoadState('loading');
    setError(null);
    try {
      const data = await apiFetch<GenerateResult>('/api/dataset/generate', {
        method: 'POST',
        body: JSON.stringify(params),
      });
      if (isMountedRef.current) {
        setGenerateResult(data);
        setLoadState('loaded');
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Dataset generation failed');
        setLoadState('error');
      }
    }
  }, []);

  const importSwing = useCallback(async (filePath: string, format?: string) => {
    setLoadState('loading');
    setError(null);
    try {
      const data = await apiFetch<GenerateResult>('/api/dataset/import-swing', {
        method: 'POST',
        body: JSON.stringify({ file_path: filePath, format }),
      });
      if (isMountedRef.current) {
        setGenerateResult(data);
        setLoadState('loaded');
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Swing import failed');
        setLoadState('error');
      }
    }
  }, []);

  const updateControl = useCallback(
    async (controlId: string, value: unknown) => {
      try {
        await apiFetch(`/api/dataset/control/${controlId}`, {
          method: 'POST',
          body: JSON.stringify({ value }),
        });
        // Refresh controls after update
        await fetchControls();
      } catch (err) {
        if (isMountedRef.current)
          setError(err instanceof Error ? err.message : 'Control update failed');
      }
    },
    [fetchControls],
  );

  /**
   * F1 — Export dataset to the chosen format.
   *
   * Triggers a browser download via a temporary anchor element so the file
   * lands in the user's Downloads folder without a separate download page.
   *
   * @param datasetId - The `dataset_id` from a `GenerateResult`
   * @param format    - Export format id (e.g. "csv", "json", "hdf5")
   */
  const exportDataset = useCallback(
    async (datasetId: string, format: string) => {
      setError(null);
      try {
        // Use raw fetch so we can read the blob response
        const { getApiBase } = await import('./backend');
        const url = `${getApiBase()}/api/dataset/export/${encodeURIComponent(datasetId)}?format=${encodeURIComponent(format)}`;
        const res = await fetch(url);
        if (!res.ok) {
          let detail: string | undefined;
          try {
            const body = (await res.json()) as Record<string, unknown>;
            if (typeof body.detail === 'string') detail = body.detail;
          } catch {
            // ignore
          }
          throw new Error(detail ?? `Export failed: HTTP ${res.status}`);
        }
        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = objectUrl;
        a.download = `${datasetId}.${format}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(objectUrl);
      } catch (err) {
        if (isMountedRef.current)
          setError(err instanceof Error ? err.message : 'Export failed');
      }
    },
    [],
  );

  // F7: Fetch catalog data on mount, set catalogLoading=false when all done
  useEffect(() => {
    isMountedRef.current = true;
    setCatalogLoading(true);
    Promise.allSettled([
      fetchFeatures(),
      fetchPlotTypes(),
      fetchExportFormats(),
      fetchControls(),
    ]).finally(() => {
      if (isMountedRef.current) setCatalogLoading(false);
    });
    return () => {
      isMountedRef.current = false;
    };
  }, [fetchFeatures, fetchPlotTypes, fetchExportFormats, fetchControls]);

  return {
    features,
    plotTypes,
    exportFormats,
    controls,
    generateResult,
    loadState,
    catalogLoading,
    error,
    generateDataset,
    importSwing,
    updateControl,
    exportDataset,
    refetch: () => {
      fetchFeatures();
      fetchPlotTypes();
      fetchExportFormats();
      fetchControls();
    },
  };
}
