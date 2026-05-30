import { renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { apiFetch } from './fetch';
import { useDatasetGenerator } from './useDatasetGenerator';

vi.mock('./fetch', () => ({
  apiFetch: vi.fn(),
}));

const feature = {
  id: 'club_speed',
  name: 'Club Speed',
  description: 'Club head speed',
  category: 'kinematics',
};

const plotType = {
  id: 'scatter',
  name: 'Scatter Plot',
  description: 'Two-variable scatter plot',
  axes: ['x', 'y'],
};

const exportFormat = {
  id: 'csv',
  name: 'CSV',
  extension: 'csv',
  mime_type: 'text/csv',
};

describe('useDatasetGenerator catalog responses', () => {
  const apiFetchMock = vi.mocked(apiFetch);

  beforeEach(() => {
    apiFetchMock.mockReset();
  });

  it('preserves bare-array catalog responses', async () => {
    apiFetchMock.mockImplementation(async (path) => {
      if (path === '/api/dataset/features') return [feature];
      if (path === '/api/dataset/plots/types') return [plotType];
      if (path === '/api/dataset/export/formats') return [exportFormat];
      if (path === '/api/dataset/control') return { controls: [] };
      throw new Error(`Unexpected path: ${path}`);
    });

    const { result } = renderHook(() => useDatasetGenerator());

    await waitFor(() => {
      expect(result.current.catalogLoading).toBe(false);
    });

    expect(result.current.features).toEqual([feature]);
    expect(result.current.plotTypes).toEqual([plotType]);
    expect(result.current.exportFormats).toEqual([exportFormat]);
  });

  it('normalizes live backend plot and export catalog records', async () => {
    apiFetchMock.mockImplementation(async (path) => {
      if (path === '/api/dataset/features') return [feature];
      if (path === '/api/dataset/plots/types') {
        return [{ type: 'histogram', description: 'Distribution plot' }];
      }
      if (path === '/api/dataset/export/formats') {
        return [{ format: 'parquet', description: 'Columnar data file' }];
      }
      if (path === '/api/dataset/control') return { controls: [] };
      throw new Error(`Unexpected path: ${path}`);
    });

    const { result } = renderHook(() => useDatasetGenerator());

    await waitFor(() => {
      expect(result.current.catalogLoading).toBe(false);
    });

    expect(result.current.plotTypes).toEqual([
      {
        id: 'histogram',
        name: 'Histogram',
        description: 'Distribution plot',
        axes: [],
      },
    ]);
    expect(result.current.exportFormats).toEqual([
      {
        id: 'parquet',
        name: 'PARQUET',
        extension: 'parquet',
        mime_type: 'Columnar data file',
      },
    ]);
  });

  it('keeps wrapper-object catalog response support', async () => {
    apiFetchMock.mockImplementation(async (path) => {
      if (path === '/api/dataset/features') return { features: [feature] };
      if (path === '/api/dataset/plots/types') return { plot_types: [plotType] };
      if (path === '/api/dataset/export/formats') return { formats: [exportFormat] };
      if (path === '/api/dataset/control') return { controls: [] };
      throw new Error(`Unexpected path: ${path}`);
    });

    const { result } = renderHook(() => useDatasetGenerator());

    await waitFor(() => {
      expect(result.current.catalogLoading).toBe(false);
    });

    expect(result.current.features).toEqual([feature]);
    expect(result.current.plotTypes).toEqual([plotType]);
    expect(result.current.exportFormats).toEqual([exportFormat]);
  });
});
