/**
 * Loop null-space exploration client and plot-model contract (NSX-44, #12576).
 *
 * The TypeScript types below mirror `NSXPlotModel.to_dict()` in
 * `src/shared/python/loop_nullspace/plot_model.py` (schema 1.0.0). The Python
 * model has already applied every display decision (ADR-0054 D7); this module
 * and the panel built on it read those decisions and compute nothing. The
 * golden `tests/fixtures/loop_nullspace/goldens/plot_model_v1.json` is the one
 * fixture both UIs are tested against.
 *
 * Job routes: `src/api/routes/analysis_loop_nullspace.py`
 * (`POST /analysis/loop-nullspace/jobs`, `GET`/`DELETE .../jobs/{id}`).
 */

import { apiFetch } from './fetch';
import type {
  LoopNullspaceJobStatusV1,
  LoopNullspaceOptions,
  LoopNullspaceRequest,
} from './generated/types';

export type { LoopNullspaceJobStatusV1, LoopNullspaceOptions, LoopNullspaceRequest };

export const PLOT_MODEL_SCHEMA_VERSION = '1.0.0';
export const LOOP_NULLSPACE_PATH = '/api/analysis/loop-nullspace';

export interface LoopNullspacePreset {
  id: string;
  label: string;
}

const isPreset = (value: unknown): value is LoopNullspacePreset =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as LoopNullspacePreset).id === 'string' &&
  typeof (value as LoopNullspacePreset).label === 'string';

/**
 * The registry presets served by `GET .../presets` (no copy is kept here).
 * Rejects a payload that is not a list of `{id, label}` strings, so a caller
 * shows an error instead of rendering a malformed list.
 */
export async function fetchLoopNullspacePresets(
  signal?: AbortSignal,
): Promise<LoopNullspacePreset[]> {
  const payload: unknown = await apiFetch<unknown>(`${LOOP_NULLSPACE_PATH}/presets`, { signal });
  if (!Array.isArray(payload) || !payload.every(isPreset)) {
    throw new Error('malformed preset list from the loop-nullspace presets endpoint');
  }
  return payload;
}

// ----------------------------------------------------------- plot model
export type DisplayAction =
  'SHOW_VALUE' | 'SHOW_BAND' | 'SHOW_REPRESENTATIVE_ON_BAND' | 'SHOW_REASON' | 'WITHHOLD' | string;

export type DisplayClass = 'DETERMINED_SIGN' | 'INDETERMINATE' | 'UNBOUNDED';

export interface DecisionView {
  action: DisplayAction;
  reason: string;
  label: string;
  badges: string[];
  hypothesis_chips: string[];
  conditioning_chips: string[];
  preset: string | null;
}

export interface WithheldItem {
  quantity_id: string;
  action: string;
  reason: string;
}

export interface Segment {
  first_frame_index: number;
  last_frame_index: number;
  display_class: DisplayClass;
  sign_class: string | null;
  binding_hypotheses: string[];
  active_constraints: string[];
}

export interface Band {
  quantity_id: string;
  units: string;
  frame_indices: number[];
  lower: Array<number | null>;
  upper: Array<number | null>;
  status_lower: string[];
  status_upper: string[];
  segments: Segment[];
  infeasible_frame_indices: number[];
  decision: DecisionView;
}

export interface FrameStatus {
  index: number;
  feasible: boolean;
  chebyshev_status: string;
  chebyshev_margin: number | null;
  conflict_hypotheses: string[];
  rests_on_unverified: boolean;
}

export interface FrameValue {
  frame_index: number;
  shown: boolean;
  values: Array<number | null>;
  decision: DecisionView | null;
  reason: string;
}

export interface InvariantSeries {
  name: string;
  units: string;
  frames: FrameValue[];
}

export interface ValueSeries {
  name: string;
  units: string;
  shown: boolean;
  values: Array<number | null>;
  decision: DecisionView | null;
  reason: string;
}

export interface Overlay {
  band_id: string;
  units: string;
  frame_indices: number[];
  values: Array<number | null>;
  decision: DecisionView;
}

export interface FrameDiagnostic {
  index: number;
  rank_A: number;
  nullity: number;
  rank_J_c: number;
  condition_number: number | null;
}

export interface AdmissionSummary {
  verdict: string;
  reasons: string[];
  failing_frames: number[];
  derivative_method: string;
  noise_floor: number | null;
  position_evaluated: boolean;
}

export interface KktSummary {
  primal: number | null;
  dual: number | null;
  complementarity: number | null;
}

export interface NSXPlotModel {
  schema_version: string;
  status: string;
  reason: string | null;
  preset: string;
  registry_version: string;
  engine_name: string;
  frame_indices: number[];
  non_claims: string[];
  invariants: { series: InvariantSeries[]; unavailable_reason: string | null };
  envelopes: {
    bands: Band[];
    withheld: WithheldItem[];
    frames: FrameStatus[];
    unavailable_reason: string | null;
  };
  hand_split: {
    frame: string;
    bands: Band[];
    withheld: WithheldItem[];
    unavailable_reason: string | null;
  };
  allocation: {
    status: string | null;
    reason: string | null;
    overlays: Overlay[];
    withheld: WithheldItem[];
    excluded_frame_indices: number[];
    objective_total: number | null;
    non_claim: string | null;
    unavailable_reason: string | null;
  };
  diagnostics: {
    frames: FrameDiagnostic[];
    admission: AdmissionSummary | null;
    residuals: ValueSeries[];
    kkt: KktSummary | null;
  };
}

const PANEL_KEYS = ['invariants', 'envelopes', 'hand_split', 'allocation', 'diagnostics'];

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/**
 * `value` as a plot model, or `null` when it is not one (fail closed).
 *
 * Only the schema version and the presence of the five panels are checked;
 * nothing is repaired or defaulted, so a model of another version is refused
 * rather than drawn.
 */
export function asPlotModel(value: unknown): NSXPlotModel | null {
  if (!isRecord(value) || value.schema_version !== PLOT_MODEL_SCHEMA_VERSION) {
    return null;
  }
  return PANEL_KEYS.every((k) => isRecord(value[k])) ? (value as unknown as NSXPlotModel) : null;
}

/**
 * The plot model a job status carries: `result.plot_model` once completed,
 * else `partial.plot_model` while running. `null` when the API supplied none;
 * the panel never builds one from the raw report.
 */
export function plotModelOf(job: LoopNullspaceJobStatusV1): NSXPlotModel | null {
  const body = job.result ?? job.partial ?? null;
  return body ? asPlotModel(body.plot_model) : null;
}

// ---------------------------------------------------------------- jobs
const JOB_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const JOB_REQUEST_TIMEOUT_MS = 15_000;

export const LOOP_NULLSPACE_POLL = {
  initialMs: 250,
  factor: 2,
  maxMs: 2_000,
} as const;

const TERMINAL: ReadonlySet<LoopNullspaceJobStatusV1['status']> = new Set([
  'completed',
  'failed',
  'cancelled',
]);

function jobPath(jobId: string): string {
  if (!JOB_ID.test(jobId)) {
    throw new Error(`"${jobId}" is not a loop-nullspace job id`);
  }
  return `${LOOP_NULLSPACE_PATH}/jobs/${jobId}`;
}

/** Submit a job (or receive a cache hit as an already completed one). */
export function submitLoopNullspaceJob(
  request: LoopNullspaceRequest,
  signal?: AbortSignal,
): Promise<LoopNullspaceJobStatusV1> {
  return apiFetch<LoopNullspaceJobStatusV1>(`${LOOP_NULLSPACE_PATH}/jobs`, {
    method: 'POST',
    body: JSON.stringify(request),
    timeoutMs: JOB_REQUEST_TIMEOUT_MS,
    signal,
  });
}

export function getLoopNullspaceJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<LoopNullspaceJobStatusV1> {
  return apiFetch<LoopNullspaceJobStatusV1>(jobPath(jobId), {
    timeoutMs: JOB_REQUEST_TIMEOUT_MS,
    signal,
  });
}

export function cancelLoopNullspaceJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<LoopNullspaceJobStatusV1> {
  return apiFetch<LoopNullspaceJobStatusV1>(jobPath(jobId), {
    method: 'DELETE',
    timeoutMs: JOB_REQUEST_TIMEOUT_MS,
    signal,
  });
}

export function isTerminalLoopNullspaceJob(job: LoopNullspaceJobStatusV1): boolean {
  return TERMINAL.has(job.status);
}

/** The wait before the next status read: backed off, never above the cap. */
export function nextLoopNullspaceDelay(previousMs: number): number {
  return Math.min(previousMs * LOOP_NULLSPACE_POLL.factor, LOOP_NULLSPACE_POLL.maxMs);
}
