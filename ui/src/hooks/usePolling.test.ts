/**
 * Tests for usePolling / useIncrementalSeries — the shared REST polling
 * hooks used by the Simulation-page panels.
 *
 * See issue #8941
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePolling } from './usePolling';
import {
  advanceCursor,
  appendSeriesPoints,
  useIncrementalSeries,
  type SeriesPage,
} from './useIncrementalSeries';

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => state,
  });
  document.dispatchEvent(new Event('visibilitychange'));
}

/** Flush resolved promises queued by an in-flight task. */
async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe('usePolling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setVisibility('visible');
  });

  afterEach(() => {
    vi.useRealTimers();
    setVisibility('visible');
  });

  it('fires once immediately and exactly once per tick', async () => {
    const task = vi.fn().mockResolvedValue(undefined);
    renderHook(() => usePolling(task, { intervalMs: 500, enabled: true }));
    await flush();
    expect(task).toHaveBeenCalledTimes(1);

    for (let tick = 1; tick <= 3; tick++) {
      await act(async () => {
        vi.advanceTimersByTime(500);
      });
      await flush();
      expect(task).toHaveBeenCalledTimes(1 + tick);
    }
  });

  it('does not overlap a tick with a still-pending request', async () => {
    let resolve: () => void = () => {};
    const task = vi.fn(
      () => new Promise<void>((r) => {
        resolve = r;
      }),
    );
    renderHook(() => usePolling(task, { intervalMs: 100, enabled: true }));
    await act(async () => {
      vi.advanceTimersByTime(350);
    });
    expect(task).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolve();
    });
    await act(async () => {
      vi.advanceTimersByTime(100);
    });
    expect(task).toHaveBeenCalledTimes(2);
  });

  it('does not poll while disabled (simulation stopped)', async () => {
    const task = vi.fn().mockResolvedValue(undefined);
    const { rerender } = renderHook(
      ({ enabled }) => usePolling(task, { intervalMs: 200, enabled }),
      { initialProps: { enabled: false } },
    );
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    expect(task).not.toHaveBeenCalled();

    rerender({ enabled: true });
    await flush();
    expect(task).toHaveBeenCalledTimes(1);

    rerender({ enabled: false });
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    expect(task).toHaveBeenCalledTimes(1);
  });

  it('pauses while the document is hidden and resumes when visible', async () => {
    const task = vi.fn().mockResolvedValue(undefined);
    renderHook(() => usePolling(task, { intervalMs: 200, enabled: true }));
    await flush();
    expect(task).toHaveBeenCalledTimes(1);

    act(() => setVisibility('hidden'));
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    expect(task).toHaveBeenCalledTimes(1);

    act(() => setVisibility('visible'));
    await flush();
    expect(task).toHaveBeenCalledTimes(2);
  });

  it('does not start while the document is hidden', async () => {
    setVisibility('hidden');
    const task = vi.fn().mockResolvedValue(undefined);
    renderHook(() => usePolling(task, { intervalMs: 200, enabled: true }));
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(task).not.toHaveBeenCalled();
  });

  it('clears its interval on unmount', async () => {
    const clearSpy = vi.spyOn(globalThis, 'clearInterval');
    const task = vi.fn().mockResolvedValue(undefined);
    const { unmount } = renderHook(() =>
      usePolling(task, { intervalMs: 200, enabled: true }),
    );
    await flush();
    unmount();
    expect(clearSpy).toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    expect(task).toHaveBeenCalledTimes(1);
    clearSpy.mockRestore();
  });

  it('uses the latest task without restarting the interval', async () => {
    const first = vi.fn().mockResolvedValue(undefined);
    const second = vi.fn().mockResolvedValue(undefined);
    const { rerender } = renderHook(
      ({ task }) => usePolling(task, { intervalMs: 200, enabled: true }),
      { initialProps: { task: first } },
    );
    await flush();
    rerender({ task: second });
    await act(async () => {
      vi.advanceTimersByTime(200);
    });
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(1);
  });

  it.each([0, -5, Number.NaN, Number.POSITIVE_INFINITY])(
    'rejects a non-positive or non-finite interval (%s)',
    (intervalMs) => {
      const task = vi.fn();
      expect(() =>
        renderHook(() => usePolling(task, { intervalMs, enabled: false })),
      ).toThrow(/intervalMs/);
    },
  );
});

describe('advanceCursor', () => {
  it('advances monotonically', () => {
    expect(advanceCursor(null, 5)).toEqual({ cursor: 5, reset: false });
    expect(advanceCursor(5, 5)).toEqual({ cursor: 5, reset: false });
    expect(advanceCursor(5, 9)).toEqual({ cursor: 9, reset: false });
  });

  it('treats a regression (server history reset) as a new epoch', () => {
    expect(advanceCursor(9, 2)).toEqual({ cursor: 2, reset: true });
  });

  it.each([-1, 1.5, Number.NaN])('rejects an invalid cursor (%s)', (next) => {
    expect(() => advanceCursor(3, next)).toThrow(/cursor/);
  });
});

describe('appendSeriesPoints', () => {
  it('appends new samples at their absolute index and keeps the window', () => {
    const first = appendSeriesPoints([], { a: [1, 2, 3], b: [10, 20, 30] }, 3, 4);
    expect(first).toEqual([
      { index: 0, a: 1, b: 10 },
      { index: 1, a: 2, b: 20 },
      { index: 2, a: 3, b: 30 },
    ]);
    const next = appendSeriesPoints(first, { a: [4, 5], b: [40, 50] }, 5, 4);
    expect(next.map((p) => p.index)).toEqual([1, 2, 3, 4]);
    expect(next[3]).toEqual({ index: 4, a: 5, b: 50 });
  });

  it('end-aligns a key that is missing from older samples', () => {
    const points = appendSeriesPoints([], { a: [1, 2], late: [9] }, 2, 10);
    expect(points).toEqual([
      { index: 0, a: 1 },
      { index: 1, a: 2, late: 9 },
    ]);
  });

  it('returns the previous array when nothing is new', () => {
    const prev = [{ index: 0, a: 1 }];
    expect(appendSeriesPoints(prev, { a: [] }, 1, 10)).toBe(prev);
  });
});

describe('useIncrementalSeries', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setVisibility('visible');
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function page(
    series: Record<string, number[]>,
    nextSince: number | null,
  ): SeriesPage<{ series: Record<string, number[]> }> {
    return { body: { series }, series, nextSince };
  }

  it('sends one request per tick and advances the since cursor from the response', async () => {
    const fetchPage = vi
      .fn()
      .mockResolvedValueOnce(page({ a: [1, 2] }, 2))
      .mockResolvedValueOnce(page({ a: [3] }, 3))
      .mockResolvedValueOnce(page({ a: [4, 5] }, 5));

    const { result } = renderHook(() =>
      useIncrementalSeries({
        fetchPage,
        maxPoints: 3,
        intervalMs: 500,
        enabled: true,
      }),
    );
    await flush();
    expect(fetchPage).toHaveBeenCalledTimes(1);
    expect(fetchPage).toHaveBeenLastCalledWith(null, 3);

    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    await flush();
    expect(fetchPage).toHaveBeenCalledTimes(2);
    expect(fetchPage).toHaveBeenLastCalledWith(2, 3);

    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    await flush();
    expect(fetchPage).toHaveBeenCalledTimes(3);
    expect(fetchPage).toHaveBeenLastCalledWith(3, 3);

    expect(result.current.points.map((p) => p.a)).toEqual([3, 4, 5]);
    expect(result.current.latest).toEqual({ series: { a: [4, 5] } });
    expect(result.current.error).toBeNull();
  });

  it('replaces the window when the server gives no cursor', async () => {
    const fetchPage = vi
      .fn()
      .mockResolvedValueOnce(page({ a: [1, 2] }, null))
      .mockResolvedValueOnce(page({ a: [1, 2, 3] }, null));
    const { result } = renderHook(() =>
      useIncrementalSeries({ fetchPage, maxPoints: 10, intervalMs: 500, enabled: true }),
    );
    await flush();
    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    await flush();
    expect(fetchPage).toHaveBeenLastCalledWith(null, 10);
    expect(result.current.points.map((p) => p.a)).toEqual([1, 2, 3]);
  });

  it('starts a new window when the cursor regresses', async () => {
    const fetchPage = vi
      .fn()
      .mockResolvedValueOnce(page({ a: [1, 2, 3] }, 3))
      .mockResolvedValueOnce(page({ a: [7] }, 1));
    const { result } = renderHook(() =>
      useIncrementalSeries({ fetchPage, maxPoints: 10, intervalMs: 500, enabled: true }),
    );
    await flush();
    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    await flush();
    expect(result.current.points).toEqual([{ index: 0, a: 7 }]);
  });

  it('surfaces fetch errors and does not poll while disabled', async () => {
    const fetchPage = vi.fn().mockRejectedValue(new Error('No engine'));
    const { result, rerender } = renderHook(
      ({ enabled }) =>
        useIncrementalSeries({ fetchPage, maxPoints: 10, intervalMs: 500, enabled }),
      { initialProps: { enabled: true } },
    );
    await flush();
    expect(result.current.error).toBe('No engine');

    rerender({ enabled: false });
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });

  it('rejects a non-positive window size', () => {
    expect(() =>
      renderHook(() =>
        useIncrementalSeries({
          fetchPage: vi.fn(),
          maxPoints: 0,
          intervalMs: 500,
          enabled: false,
        }),
      ),
    ).toThrow(/maxPoints/);
  });
});
