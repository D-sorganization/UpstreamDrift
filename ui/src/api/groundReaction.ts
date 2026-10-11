/**
 * Ground-reaction client (GCV-5, #11711).
 *
 * GET /api/analysis/ground-reaction returns the ground-reaction time series
 * on the body: force per foot and net (in body weights too when a subject
 * weight is known), centre of pressure, free moment at the centre of
 * pressure and moment about the whole-body centre of mass, plus each foot's
 * share of the summed vertical force. A `null` sample is unavailable and
 * must be drawn as a gap, never as zero.
 */

import { apiFetch } from './fetch';
import type { GroundReactionResponse } from './generated/types';

export { availableRange, pathSegments } from './traceGeometry';

export type { GroundReactionResponse };

export interface GroundReactionQuery {
  runId?: string | null;
  impactTimeS?: number | null;
}

export function buildGroundReactionPath(q: GroundReactionQuery): string {
  const params = new URLSearchParams();
  if (q.runId) params.set('run_id', q.runId);
  if (q.impactTimeS !== null && q.impactTimeS !== undefined) {
    params.set('impact_time_s', String(q.impactTimeS));
  }
  const text = params.toString();
  return `/api/analysis/ground-reaction${text ? `?${text}` : ''}`;
}

export function fetchGroundReaction(
  q: GroundReactionQuery,
  signal?: AbortSignal,
): Promise<GroundReactionResponse> {
  return apiFetch<GroundReactionResponse>(buildGroundReactionPath(q), { signal });
}
