/**
 * TDD tests for useDatasetGenerator catalog fetching.
 *
 * Regression for review-feedback #6703: the dataset catalog endpoints
 * (`/api/dataset/features`, `/plots/types`, `/export/formats`) return BARE
 * ARRAYS from the backend (src/api/routes/dataset.py). The hook must populate
 * its state from those bare arrays, while still tolerating a future
 * `{features: [...]}` / `{plot_types: [...]}` / `{formats: [...]}` wrapper.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useDatasetGenerator } from './useDatasetGenerator';
import type { FeatureInfo, PlotType, ExportFormat } from './useDatasetGenerator';

const FEATURES: FeatureInfo[] = [
  { id: 'f1', name: 'Feature One', description: 'desc', category: 'control' },
];
const PLOT_TYPES: PlotType[] = [
  { id: 'scatter', name: 'Scatter', description: 'desc', axes: ['x', 'y'] },
];
const EXPORT_FORMATS: ExportFormat[] = [
  { id: 'csv', name: 'CSV', extension: '.csv', mime_type: 'text/csv' },
];

const originalFetch = global.fetch;

/** Route a mocked fetch by URL suffix to the supplied JSON payloads. */
function mockCatalog(payloads: {
  features: unknown;
  plots: unknown;
  formats: unknown;
  controls?: unknown;
}) {
  global.fetch = vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    const pick = url.endsWith('/api/dataset/features')
      ? payloads.features
      : url.endsWith('/api/dataset/plots/types')
        ? payloads.plots
        : url.endsWith('/api/dataset/export/formats')
          ? payloads.formats
          : (payloads.controls ?? { controls: [] });
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve(pick),
    } as Response);
  }) as typeof fetch;
}

describe('useDatasetGenerator catalog parsing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('populates catalogs from BARE-ARRAY backend responses (#6703)', async () => {
    mockCatalog({ features: FEATURES, plots: PLOT_TYPES, formats: EXPORT_FORMATS });

    const { result } = renderHook(() => useDatasetGenerator());

    await waitFor(() => expect(result.current.catalogLoading).toBe(false));

    expect(result.current.features).toEqual(FEATURES);
    expect(result.current.plotTypes).toEqual(PLOT_TYPES);
    expect(result.current.exportFormats).toEqual(EXPORT_FORMATS);
    expect(result.current.error).toBeNull();
  });

  it('still tolerates wrapped-object responses', async () => {
    mockCatalog({
      features: { features: FEATURES },
      plots: { plot_types: PLOT_TYPES },
      formats: { formats: EXPORT_FORMATS },
    });

    const { result } = renderHook(() => useDatasetGenerator());

    await waitFor(() => expect(result.current.catalogLoading).toBe(false));

    expect(result.current.features).toEqual(FEATURES);
    expect(result.current.plotTypes).toEqual(PLOT_TYPES);
    expect(result.current.exportFormats).toEqual(EXPORT_FORMATS);
  });

  it('falls back to empty arrays on malformed payloads', async () => {
    mockCatalog({ features: 'nope', plots: 42, formats: null });

    const { result } = renderHook(() => useDatasetGenerator());

    await waitFor(() => expect(result.current.catalogLoading).toBe(false));

    expect(result.current.features).toEqual([]);
    expect(result.current.plotTypes).toEqual([]);
    expect(result.current.exportFormats).toEqual([]);
  });
});
