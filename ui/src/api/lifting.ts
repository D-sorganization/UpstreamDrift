/**
 * LIFT-1 cross-engine baseline API client (LIFT-8 slice 3, #11748).
 *
 * Typed client for the read-only baseline routes in
 * `src/api/routes/lifting.py`, backed by
 * `src/shared/python/lifting/baseline_view.py`. Every scalar measurement is
 * returned as a `NumericField` so a missing or non-finite receipt value can
 * never be silently read as a real zero ("unavailable is never zero").
 */

import { getApiBase } from './backend';

/** A scalar receipt measurement: present-and-finite, or explained-absent. */
export interface NumericField {
  value: number | null;
  reason: string | null;
}

/** A cross-engine pairwise position metric, flagged against the tolerance. */
export interface PairMetric extends NumericField {
  status: 'pass' | 'fail' | 'unavailable';
  tolerance_m: number;
}

export type PairMetricKey =
  | 'segments_max_m'
  | 'hands_max_m'
  | 'feet_max_m'
  | 'bar_centre_max_m'
  | 'com_max_m'
  | 'lifter_com_max_m';

/** Ordered for stable table columns; mirrors `baseline_view._PAIR_METRICS`. */
export const PAIR_METRICS: readonly PairMetricKey[] = [
  'segments_max_m',
  'hands_max_m',
  'feet_max_m',
  'bar_centre_max_m',
  'com_max_m',
  'lifter_com_max_m',
];

export type PairMetrics = Record<PairMetricKey, PairMetric>;

export interface PackInfo {
  repo: string | null;
  commit: string | null;
  licence: string | null;
}

export interface StructureInfo {
  n_bodies: number | null;
  nq: number | null;
  nv: number | null;
}

export interface SmokeInfo {
  loaded: boolean | null;
  stepped: boolean | null;
  max_abs_qvel: NumericField;
}

export interface StartContactView {
  value_n: NumericField;
  non_ground_normal_force_n: NumericField;
  n_ground_contacts: number | null;
  n_non_ground_contacts: number | null;
  reason: string | null;
}

export interface HandBarAxisDistance {
  l: NumericField;
  r: NumericField;
}

export interface PhaseView {
  name: string | null;
  fraction: NumericField;
  n_targets: number | null;
  hand_bar_axis_distance_m: HandBarAxisDistance;
}

export interface EngineView {
  engine: string;
  pack: PackInfo;
  structure: StructureInfo;
  total_mass_kg: NumericField;
  bar_above_sole_m: NumericField;
  hand_mid_above_sole_m: NumericField;
  smoke: SmokeInfo;
  start_contact: StartContactView;
  phases: PhaseView[];
}

export interface ComparisonsView {
  poses: Record<string, Record<string, PairMetrics>>;
  reason: string | null;
}

export interface GapEntry {
  key: string;
  title: string;
  engines: string[];
  evidence: string[];
  issues: string[];
  lift_story: string;
  new_issue: boolean;
}

export interface LiftView {
  lift: string;
  engines: EngineView[];
  comparisons: ComparisonsView;
  gaps: GapEntry[];
}

export interface Tolerances {
  position_m: number;
  mass_rel: number;
}

export interface LiftBaselineMetadata {
  schema: string;
  generated_utc: string;
  anthropometry: Record<string, number>;
  tolerances: Tolerances;
  packs: Record<string, PackInfo>;
  lifts: string[];
  gap_count: number;
}

function apiUrl(path: string): string {
  return `${getApiBase()}${path}`;
}

export async function fetchLiftBaseline(): Promise<LiftBaselineMetadata> {
  const resp = await fetch(apiUrl('/api/v1/lifting/baseline'));
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load lift baseline: ${resp.status} ${text}`);
  }
  return resp.json();
}

export async function fetchLiftBaselineLift(lift: string): Promise<LiftView> {
  const resp = await fetch(
    apiUrl(`/api/v1/lifting/baseline/lifts/${encodeURIComponent(lift)}`),
  );
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load lift view for ${lift}: ${resp.status} ${text}`);
  }
  return resp.json();
}

/**
 * Format a `NumericField` for display. "Unavailable is never zero": a
 * missing or non-finite value renders as the literal string `'unavailable'`
 * rather than `0`; callers that want to surface *why* should read
 * `field.reason` themselves (e.g. as a `title` attribute).
 */
export function formatMeasurement(
  field: NumericField | null | undefined,
  digits: number,
  unit: string,
): string {
  if (!field || field.value == null) {
    return 'unavailable';
  }
  const formatted = field.value.toFixed(digits);
  return unit ? `${formatted} ${unit}` : formatted;
}
