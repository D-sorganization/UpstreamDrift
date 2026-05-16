/**
 * Dataset Generator Hook — Fetches dataset generation controls, features,
 * plot types, and export formats from the backend REST API.
 *
 * Provides methods for generating datasets, importing swing data, and
 * managing dataset control parameters.
 */

import { useState, useCallback, useEffect, useRef } from 'react';

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

// ── Hook ───────────────────────────────────────────────────────────────

export function useDatasetGenerator() {
  const [features, setFeatures] = useState<FeatureInfo[]>([]);
  const [plotTypes, setPlotTypes] = useState<PlotType[]>([]);
  const [exportFormats, setExportFormats] = useState<ExportFormat[]>([]);
  const [controls, setControls] = useState<DatasetControl[]>([]);
  const [generateResult, setGenerateResult] = useState<GenerateResult | null>(null);
  const [loadState, setLoadState] = useState<DatasetLoadState>('idle');
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  const fetchFeatures = useCallback(async () => {
    try {
      const res = await fetch('/api/dataset/features');
      if (!res.ok) throw new Error(`Failed to fetch features: ${res.status}`);
      const data = await res.json();
      if (isMountedRef.current) setFeatures(data.features ?? data ?? []);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch features');
    }
  }, []);

  const fetchPlotTypes = useCallback(async () => {
    try {
      const res = await fetch('/api/dataset/plots/types');
      if (!res.ok) throw new Error(`Failed to fetch plot types: ${res.status}`);
      const data = await res.json();
      if (isMountedRef.current) setPlotTypes(data.plot_types ?? data.types ?? data ?? []);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch plot types');
    }
  }, []);

  const fetchExportFormats = useCallback(async () => {
    try {
      const res = await fetch('/api/dataset/export/formats');
      if (!res.ok) throw new Error(`Failed to fetch export formats: ${res.status}`);
      const data = await res.json();
      if (isMountedRef.current) setExportFormats(data.formats ?? data ?? []);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch export formats');
    }
  }, []);

  const fetchControls = useCallback(async () => {
    try {
      const res = await fetch('/api/dataset/control');
      if (!res.ok) throw new Error(`Failed to fetch controls: ${res.status}`);
      const data = await res.json();
      if (isMountedRef.current) setControls(data.controls ?? data ?? []);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch controls');
    }
  }, []);

  const generateDataset = useCallback(async (params: Record<string, unknown>) => {
    setLoadState('loading');
    setError(null);
    try {
      const res = await fetch('/api/dataset/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Generation failed: ${res.status}`);
      }
      const data = await res.json();
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
      const res = await fetch('/api/dataset/import-swing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: filePath, format }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Import failed: ${res.status}`);
      }
      const data = await res.json();
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

  const updateControl = useCallback(async (controlId: string, value: unknown) => {
    try {
      const res = await fetch(`/api/dataset/control/${controlId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value }),
      });
      if (!res.ok) throw new Error(`Control update failed: ${res.status}`);
      // Refresh controls after update
      await fetchControls();
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Control update failed');
    }
  }, [fetchControls]);

  // Fetch catalog data on mount
  useEffect(() => {
    isMountedRef.current = true;
    fetchFeatures();
    fetchPlotTypes();
    fetchExportFormats();
    fetchControls();
    return () => { isMountedRef.current = false; };
  }, [fetchFeatures, fetchPlotTypes, fetchExportFormats, fetchControls]);

  return {
    features,
    plotTypes,
    exportFormats,
    controls,
    generateResult,
    loadState,
    error,
    generateDataset,
    importSwing,
    updateControl,
    refetch: () => { fetchFeatures(); fetchPlotTypes(); fetchExportFormats(); fetchControls(); },
  };
}
