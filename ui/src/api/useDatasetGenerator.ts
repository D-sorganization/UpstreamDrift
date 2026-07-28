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
import { triggerBlobDownload } from './download';
import { apiFetch, apiFetchBlob } from './fetch';

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

function catalogArray<T>(data: unknown, keys: string[]): T[] {
  if (Array.isArray(data)) {
    return data;
  }
  if (!data || typeof data !== 'object') {
    return [];
  }

  const record = data as Record<string, unknown>;
  for (const key of keys) {
    const value = record[key];
    if (Array.isArray(value)) {
      return value;
    }
  }
  return [];
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : null;
}

function stringField(record: Record<string, unknown>, key: string): string | null {
  const value = record[key];
  return typeof value === 'string' && value.trim() ? value : null;
}

function labelFromId(id: string): string {
  return id
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function normalizePlotType(value: unknown): PlotType | null {
  const record = asRecord(value);
  if (!record) return null;

  const id = stringField(record, 'id') ?? stringField(record, 'type');
  if (!id) return null;

  const axes = Array.isArray(record.axes)
    ? record.axes.filter((axis): axis is string => typeof axis === 'string')
    : [];

  return {
    id,
    name: stringField(record, 'name') ?? labelFromId(id),
    description: stringField(record, 'description') ?? '',
    axes,
  };
}

function normalizeExportFormat(value: unknown): ExportFormat | null {
  const record = asRecord(value);
  if (!record) return null;

  const id = stringField(record, 'id') ?? stringField(record, 'format');
  if (!id) return null;

  const extension = stringField(record, 'extension') ?? id.replace(/^\./, '');
  const name = stringField(record, 'name') ?? extension.toUpperCase();

  return {
    id,
    name,
    extension,
    mime_type: stringField(record, 'mime_type') ?? stringField(record, 'description') ?? '',
  };
}

function normalizeCatalog<T>(
  data: unknown,
  keys: string[],
  normalize: (value: unknown) => T | null,
): T[] {
  return catalogArray<unknown>(data, keys).map(normalize).filter((value): value is T => value !== null);
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
      const list = catalogArray<FeatureInfo>(data, ['features']);
      if (isMountedRef.current) setFeatures(list);
    } catch (err) {
      if (isMountedRef.current)
        setError(err instanceof Error ? err.message : 'Failed to fetch features');
    }
  }, []);

  const fetchPlotTypes = useCallback(async () => {
    try {
      const data = await apiFetch<unknown>(
        '/api/dataset/plots/types',
      );
      const list = normalizeCatalog<PlotType>(data, ['plot_types', 'types'], normalizePlotType);
      if (isMountedRef.current) setPlotTypes(list);
    } catch (err) {
      if (isMountedRef.current)
        setError(err instanceof Error ? err.message : 'Failed to fetch plot types');
    }
  }, []);

  const fetchExportFormats = useCallback(async () => {
    try {
      const data = await apiFetch<unknown>(
        '/api/dataset/export/formats',
      );
      const list = normalizeCatalog<ExportFormat>(data, ['formats'], normalizeExportFormat);
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

  /**
   * Issue #7981 — control values are client-side until generation.
   *
   * This used to POST `/api/dataset/control/{id}`, an endpoint that has never
   * existed on the backend, so every keystroke produced a 404 and an error
   * banner. Generation parameters are collected locally and submitted as a
   * single body to `POST /api/dataset/generate`; there is no per-control
   * server-side state to update.
   */
  const updateControl = useCallback((controlId: string, value: unknown) => {
    setControls((prev) =>
      prev.map((ctrl) => (ctrl.id === controlId ? { ...ctrl, value } : ctrl)),
    );
  }, []);

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
        const blob = await apiFetchBlob(
          `/api/dataset/export/${encodeURIComponent(datasetId)}?format=${encodeURIComponent(format)}`,
        );
        triggerBlobDownload(blob, `${datasetId}.${format}`);
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
    queueMicrotask(() => {
      Promise.allSettled([
        fetchFeatures(),
        fetchPlotTypes(),
        fetchExportFormats(),
        fetchControls(),
      ]).finally(() => {
        if (isMountedRef.current) setCatalogLoading(false);
      });
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
