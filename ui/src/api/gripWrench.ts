/**
 * Grip wrench client (GCV-10, #11716).
 *
 * GET /api/analysis/grip-wrench returns the hands' loading on the club: force
 * per hand, net force at the grip midpoint and the equivalent couple about the
 * midpoint (world and club frames), with `split_method`. A `null` sample is
 * unavailable and must be drawn as a gap, never as zero.
 */

import { apiFetch } from './fetch';
import type { GripWrenchResponse } from './generated/types';

export type { GripWrenchResponse };

export type GripTraceName =
  | 'left_force_n'
  | 'right_force_n'
  | 'net_force_n'
  | 'couple_nm'
  | 'couple_local_nm'
  | 'contact_force_moment_nm'
  | 'applied_free_torque_nm'
  | 'mof_left_nm'
  | 'mof_right_nm';

export interface GripWrenchQuery {
  runId?: string | null;
  impactTimeS?: number | null;
}

export function buildGripWrenchPath(q: GripWrenchQuery): string {
  const params = new URLSearchParams();
  if (q.runId) params.set('run_id', q.runId);
  if (q.impactTimeS !== null && q.impactTimeS !== undefined) {
    params.set('impact_time_s', String(q.impactTimeS));
  }
  const text = params.toString();
  return `/api/analysis/grip-wrench${text ? `?${text}` : ''}`;
}

export function fetchGripWrench(
  q: GripWrenchQuery,
  signal?: AbortSignal,
): Promise<GripWrenchResponse> {
  return apiFetch<GripWrenchResponse>(buildGripWrenchPath(q), { signal });
}

/** Runs of consecutive available samples as SVG path data; null breaks the line. */
export function pathSegments(
  t: number[],
  y: Array<number | null>,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  xRange: [number, number],
  yRange: [number, number],
): string[] {
  const sx = (v: number) =>
    x0 + ((v - xRange[0]) / (xRange[1] - xRange[0] || 1)) * (x1 - x0);
  const sy = (v: number) =>
    y1 - ((v - yRange[0]) / (yRange[1] - yRange[0] || 1)) * (y1 - y0);
  const out: string[] = [];
  let cur = '';
  y.forEach((v, i) => {
    if (v === null || v === undefined || !Number.isFinite(v)) {
      if (cur) out.push(cur);
      cur = '';
      return;
    }
    cur += `${cur ? 'L' : 'M'}${sx(t[i]).toFixed(2)},${sy(v).toFixed(2)}`;
  });
  if (cur) out.push(cur);
  return out;
}

/** Range of the available samples across series; `null` when none are available. */
export function availableRange(series: Array<Array<number | null>>): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  series.forEach((s) =>
    s.forEach((v) => {
      if (v !== null && v !== undefined && Number.isFinite(v)) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }),
  );
  return Number.isFinite(lo) ? [lo, hi] : null;
}
