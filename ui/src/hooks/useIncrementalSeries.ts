/**
 * useIncrementalSeries — cursor-based time-series polling on top of
 * {@link usePolling} (issue #8941).
 *
 * Each tick issues ONE request for the samples after the last cursor the
 * server reported, appends them, and keeps only the most recent `maxPoints`.
 * The first request (and any request after a cursor reset) asks the server
 * for at most `maxPoints` samples, so the full history is never re-sent.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { usePolling } from './usePolling';

/** One chart row: the absolute sample index plus a value per metric. */
export type SeriesPoint = Record<string, number> & { index: number };

/** What `fetchPage` returns for one tick. */
export interface SeriesPage<T> {
  /** The full response body, exposed to the caller as `latest`. */
  body: T;
  /** New samples per metric (oldest first), or null when there are none. */
  series: Record<string, number[]> | null;
  /**
   * Absolute index of the next sample the server will store, or null when
   * the server does not support incremental fetches (the window is then
   * replaced on every tick instead of appended to).
   */
  nextSince: number | null;
}

export interface UseIncrementalSeriesOptions<T> {
  /** Fetch samples at absolute index >= `since` (null = latest window). */
  fetchPage: (since: number | null, limit: number) => Promise<SeriesPage<T>>;
  /** Size of the retained window. Contract: an integer > 0. */
  maxPoints: number;
  intervalMs: number;
  enabled: boolean;
}

export interface IncrementalSeries<T> {
  /** Most recent response body, or null before the first success. */
  latest: T | null;
  /** Retained chart rows, oldest first, at most `maxPoints` long. */
  points: SeriesPoint[];
  /** Last fetch error message, or null after a success. */
  error: string | null;
}

/**
 * Advance the `since` cursor.
 *
 * Invariant: within one series epoch the cursor never decreases. A lower
 * value from the server means its history was reset (e.g. a restart), so a
 * new epoch starts and the caller must drop the points it holds.
 *
 * @throws RangeError when `next` is not a non-negative integer.
 */
export function advanceCursor(
  prev: number | null,
  next: number,
): { cursor: number; reset: boolean } {
  if (!Number.isInteger(next) || next < 0) {
    throw new RangeError(`series cursor must be a non-negative integer (got ${next})`);
  }
  return { cursor: next, reset: prev !== null && next < prev };
}

/**
 * Append new samples (ending just before absolute index `nextSince`) to the
 * chart rows, keeping the last `maxPoints`. Metrics are end-aligned, so a
 * metric that appears later still lines up with the newest samples.
 */
export function appendSeriesPoints(
  prev: SeriesPoint[],
  series: Record<string, number[]>,
  nextSince: number,
  maxPoints: number,
): SeriesPoint[] {
  const count = Math.max(0, ...Object.values(series).map((values) => values.length));
  if (count === 0) {
    return prev;
  }
  const first = nextSince - count;
  const fresh: SeriesPoint[] = [];
  for (let offset = 0; offset < count; offset++) {
    fresh.push({ index: first + offset } as SeriesPoint);
  }
  for (const [key, values] of Object.entries(series)) {
    const shift = count - values.length;
    values.forEach((value, i) => {
      fresh[shift + i][key] = value;
    });
  }
  return [...prev, ...fresh].slice(-maxPoints);
}

function assertValidMaxPoints(maxPoints: number): void {
  if (!Number.isInteger(maxPoints) || maxPoints <= 0) {
    throw new RangeError(
      `useIncrementalSeries: maxPoints must be an integer > 0 (got ${maxPoints})`,
    );
  }
}

export function useIncrementalSeries<T>({
  fetchPage,
  maxPoints,
  intervalMs,
  enabled,
}: UseIncrementalSeriesOptions<T>): IncrementalSeries<T> {
  assertValidMaxPoints(maxPoints);

  const [latest, setLatest] = useState<T | null>(null);
  const [points, setPoints] = useState<SeriesPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const cursorRef = useRef<number | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const tick = useCallback(async () => {
    try {
      const page = await fetchPage(cursorRef.current, maxPoints);
      if (!mountedRef.current) {
        return;
      }
      setLatest(page.body);
      setError(null);
      const series = page.series ?? {};
      if (page.nextSince === null) {
        cursorRef.current = null;
        const total = Math.max(0, ...Object.values(series).map((v) => v.length));
        setPoints(appendSeriesPoints([], series, total, maxPoints));
        return;
      }
      const { cursor, reset } = advanceCursor(cursorRef.current, page.nextSince);
      cursorRef.current = cursor;
      setPoints((prev) => appendSeriesPoints(reset ? [] : prev, series, cursor, maxPoints));
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch');
      }
    }
  }, [fetchPage, maxPoints]);

  usePolling(tick, { intervalMs, enabled });

  return { latest, points, error };
}
