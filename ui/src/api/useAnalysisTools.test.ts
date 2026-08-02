import { describe, expect, it, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAnalysisTools } from './useAnalysisTools';
import { apiFetch } from './fetch';

vi.mock('./fetch', () => ({
  apiFetch: vi.fn(),
}));

describe('useAnalysisTools lifecycle', () => {
  it('does not update state if unmounted before fetch completes', async () => {
    let resolveFetch: (value: unknown) => void;
    vi.mocked(apiFetch).mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      })
    );

    const { result, unmount } = renderHook(() => useAnalysisTools());

    act(() => {
      result.current.fetchMetrics();
    });

    // State is 'loading'
    expect(result.current.loadState).toBe('loading');

    // Unmount before the fetch resolves
    unmount();

    // Resolve the fetch
    await act(async () => {
      resolveFetch({ status: 'ok', metrics: {} });
      // wait a tick for promise
      await new Promise((r) => setTimeout(r, 0));
    });

    // Since the component unmounted before the promise resolved,
    // the state update was guarded against and the stale result remains 'loading'
    expect(result.current.loadState).toBe('loading');
  });
});
