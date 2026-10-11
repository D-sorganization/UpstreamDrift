/**
 * Impact parameters client (GCV-17, #11723).
 *
 * GET /api/analysis/impact-parameters returns a launch-monitor card for a run.
 * A row with `value === null` is unavailable and carries a `reason`; the UI
 * must never render it as zero.
 */

import { apiFetch } from './fetch';
import type {
  ImpactParameterRow,
  ImpactParametersResponse,
} from './generated/types';

export type { ImpactParameterRow, ImpactParametersResponse };

export interface ImpactParametersQuery {
  runId?: string | null;
  /** Target heading as a horizontal "x,y" direction (default -Y). */
  targetDir?: string | null;
  handedness?: 'right' | 'left';
  units?: 'mph' | 'm/s';
}

export function buildImpactParametersPath(q: ImpactParametersQuery): string {
  const params = new URLSearchParams();
  if (q.runId) params.set('run_id', q.runId);
  if (q.targetDir) params.set('target_dir', q.targetDir);
  params.set('handedness', q.handedness ?? 'right');
  params.set('units', q.units ?? 'mph');
  return `/api/analysis/impact-parameters?${params.toString()}`;
}

/** Target direction "x,y" from a heading in degrees (CCW from the default -Y). */
export function targetDirFromHeading(headingDeg: number): string {
  const rad = (headingDeg * Math.PI) / 180;
  const x = Math.sin(rad);
  const y = -Math.cos(rad);
  return `${x.toFixed(6)},${y.toFixed(6)}`;
}

export function fetchImpactParameters(
  q: ImpactParametersQuery,
  signal?: AbortSignal,
): Promise<ImpactParametersResponse> {
  return apiFetch<ImpactParametersResponse>(buildImpactParametersPath(q), { signal });
}

export function formatImpactValue(row: ImpactParameterRow): string {
  if (row.value === null || row.value === undefined) return 'unavailable';
  const text = `${row.value.toFixed(1)} ${row.unit}`.trim();
  return row.note ? `${text} (${row.note})` : text;
}

/** Link to the Impact Explorer prefilled with the delivery (available values only). */
export function impactExplorerHref(rows: ImpactParameterRow[]): string {
  const params = new URLSearchParams();
  rows.forEach((r) => {
    if (r.value !== null && r.value !== undefined) params.set(r.key, r.value.toFixed(3));
  });
  const query = params.toString();
  return query ? `/tools/impact-explorer?${query}` : '/tools/impact-explorer';
}
