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

export { availableRange, bandSegments, pathSegments } from './traceGeometry';

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

/** Plate facts a swing file cannot carry (mirrors the API `PlateDeclarationsModel`). */
export interface PlateDeclarations {
  force_floor_n: number;
  force_fraction: number;
  cop_m: number;
  sensing_depth_sigma_m: number;
  uncertainty_source: string;
  sensing_depth_m: number;
  capacity_n: number;
  device?: string | null;
}

/** Plate clock to model clock mapping (mirrors the API `PlateAlignmentModel`). */
export interface PlateAlignment {
  offset_s: number;
  sigma_offset_s: number;
  kind: 'absolute_utc' | 'shot_impact' | 'manual' | 'cross_correlation';
  note: string;
  derived_from_model?: string | null;
}

/**
 * POST /api/analysis/ground-reaction/plate-overlay (OF-07, #13087). `trial` is
 * a swing `<id>.json` relative to the server's plate-trial root; the server
 * refuses anything outside it. Shear is `null` in the measured traces (a
 * vertical plate does not observe it) and `measured_qualified === false`
 * marks a measurement with no device qualification.
 */
export interface PlateOverlayRequest {
  run_id?: string | null;
  impact_time_s?: number | null;
  trial: string;
  handedness: 'RH' | 'LH';
  origin_m: [number, number, number];
  target_dir?: [number, number, number];
  declared: PlateDeclarations;
  alignment: PlateAlignment;
  coverage?: number;
}

export const PLATE_OVERLAY_PATH = '/api/analysis/ground-reaction/plate-overlay';

export function fetchGroundReactionOverlay(
  request: PlateOverlayRequest,
  signal?: AbortSignal,
): Promise<GroundReactionResponse> {
  return apiFetch<GroundReactionResponse>(PLATE_OVERLAY_PATH, {
    method: 'POST',
    body: JSON.stringify(request),
    signal,
  });
}
