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

export { availableRange, pathSegments } from './traceGeometry';

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
