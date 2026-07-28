import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { downloadAnalysisExport } from './analysisExport';
import type { ExportResult } from './analysisExport';
import { apiFetch } from './fetch';
import { useAnalysisTools } from './useAnalysisTools';

vi.mock('./fetch', () => ({
  apiFetch: vi.fn(),
}));

vi.mock('./analysisExport', () => ({
  EXPORT_FORMATS: ['csv', 'json'],
  downloadAnalysisExport: vi.fn(),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

describe('useAnalysisTools lifecycle guard', () => {
  const apiFetchMock = vi.mocked(apiFetch);
  const downloadAnalysisExportMock = vi.mocked(downloadAnalysisExport);

  beforeEach(() => {
    apiFetchMock.mockReset();
    downloadAnalysisExportMock.mockReset();
  });

  it('does not commit metrics state after an in-flight request resolves post-unmount', async () => {
    const pending = deferred<unknown>();
    apiFetchMock.mockReturnValueOnce(pending.promise);
    const { result, unmount } = renderHook(() => useAnalysisTools());

    let request!: Promise<void>;
    act(() => {
      request = result.current.fetchMetrics();
    });
    expect(result.current.loadState).toBe('loading');

    unmount();
    await act(async () => {
      pending.resolve({ status: 'ok', metrics: { club_speed: 42 } });
      await request;
    });

    expect(result.current.metrics).toBeNull();
    expect(result.current.loadState).toBe('loading');
  });

  it('does not commit export state after an in-flight export resolves post-unmount', async () => {
    const pending = deferred<ExportResult>();
    downloadAnalysisExportMock.mockReturnValueOnce(pending.promise);
    const { result, unmount } = renderHook(() => useAnalysisTools());

    let request!: Promise<void>;
    act(() => {
      request = result.current.exportAnalysis('csv');
    });
    expect(result.current.loadState).toBe('loading');

    unmount();
    await act(async () => {
      pending.resolve({
        format: 'csv',
        filename: 'analysis_export.csv',
        size_bytes: 10,
      });
      await request;
    });

    expect(result.current.exportResult).toBeNull();
    expect(result.current.loadState).toBe('loading');
  });
});
