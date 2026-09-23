/**
 * Incremental fetch of `GET /api/analysis/statistics` (issue #8941).
 *
 * One request per polling tick: `collect=true` makes the server store a fresh
 * metric snapshot before aggregating (replacing the old separate
 * `/api/analysis/metrics` call), `since` asks only for samples the client has
 * not seen, and the next cursor comes back in `X-Analysis-Next-Since`.
 */

import { apiFetchWithHeaders } from './fetch';
import type { SeriesPage } from '@/hooks/useIncrementalSeries';

/** Response header carrying the absolute index of the next stored sample. */
export const NEXT_SINCE_HEADER = 'X-Analysis-Next-Since';

/** Server-side upper bound for `limit` (`MAX_METRIC_HISTORY`). */
const MAX_SERVER_LIMIT = 500;

/** Analysis metric summary from the backend. */
export interface AnalysisMetric {
  metric_name: string;
  current: number;
  minimum: number;
  maximum: number;
  mean: number;
  std_dev: number;
}

/** Statistics response from the backend. */
export interface AnalysisStatistics {
  sim_time: number;
  sample_count: number;
  metrics: AnalysisMetric[];
  time_series: Record<string, number[]> | null;
}

/** Parse the cursor header; null when absent or malformed (older server). */
export function parseNextSince(headers: Headers | undefined): number | null {
  const raw = headers?.get(NEXT_SINCE_HEADER);
  if (raw === null || raw === undefined || raw.trim() === '') {
    return null;
  }
  const value = Number(raw);
  return Number.isInteger(value) && value >= 0 ? value : null;
}

/**
 * Collect a snapshot and fetch the statistics page after `since`.
 *
 * @param since - Absolute sample index from the previous page, or null.
 * @param limit - Maximum number of recent samples to return (1..500).
 */
export async function fetchAnalysisStatisticsPage(
  since: number | null,
  limit: number,
): Promise<SeriesPage<AnalysisStatistics>> {
  const serverLimit = Math.min(Math.max(1, Math.floor(limit)), MAX_SERVER_LIMIT);
  const params = new URLSearchParams({ collect: 'true', limit: String(serverLimit) });
  if (since !== null) {
    params.set('since', String(since));
  }
  const { data, headers } = await apiFetchWithHeaders<AnalysisStatistics>(
    `/api/analysis/statistics?${params}`,
  );
  return { body: data, series: data.time_series, nextSince: parseNextSince(headers) };
}
