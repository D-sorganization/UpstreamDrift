/**
 * Loop null-space exploration client and plot-model contract (NSX-44, #12576).
 *
 * The TypeScript types below mirror `NSXPlotModel.to_dict()` in
 * `src/shared/python/loop_nullspace/plot_model.py` (schema 1.0.0, and 2.0.0
 * with the `sensitivity` panels of NSX-99 / NSX-100 #12729). The Python
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

/** A disturbance as the API's `DisturbanceModel` takes it (NSX-99). */
export interface DisturbancePayload {
  name: string;
  stage: string;
  channels: string[];
  units: string;
  covariance: number[][];
  spectrum: string | null;
  persistent: boolean;
  bandwidth_hz: number | null;
  band_hz: [number, number] | null;
  signal_gain: number[] | null;
  signal_bounds: Array<[number, number]> | null;
}

/** The API's `SensitivityOptions` (NSX-99); every id comes from the catalogue. */
export interface SensitivityOptionsPayload {
  disturbance_spec: DisturbancePayload[];
  mechanisms: string[];
  strategy_set: string[] | null;
  outcome_priority: string[] | null;
}

/** A job request; `options.sensitivity` is newer than the generated types. */
export type LoopNullspaceJobRequest = Omit<LoopNullspaceRequest, 'options'> & {
  options?: LoopNullspaceOptions & { sensitivity?: SensitivityOptionsPayload };
};

export const PLOT_MODEL_SCHEMA_VERSION = '1.0.0';
/** Plot model version that adds the `sensitivity` panels (NSX-99). */
export const PLOT_MODEL_SENSITIVITY_VERSION = '2.0.0';
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

// ------------------------------------------------- sensitivity catalogue
export interface CatalogueEntry {
  id: string;
  label: string;
}

export interface StrategyEntry {
  id: string;
  name: string;
  mechanism: string;
  availability: string;
  reason: string;
}

export interface EngineEntry {
  id: string;
  method: string;
  available: boolean;
  reason: string;
}

/** `GET .../sensitivity-options`: every value a sensitivity request may name. */
export interface SensitivityCatalogue {
  stages: CatalogueEntry[];
  spectra: CatalogueEntry[];
  mechanisms: Array<CatalogueEntry & { note: string }>;
  strategies: StrategyEntry[];
  outcomes: CatalogueEntry[];
  default_outcome_priority: string[];
  engines: EngineEntry[];
}

const CATALOGUE_LISTS = [
  'stages',
  'spectra',
  'mechanisms',
  'strategies',
  'outcomes',
  'engines',
] as const;

const isCatalogue = (v: unknown): v is SensitivityCatalogue =>
  typeof v === 'object' &&
  v !== null &&
  CATALOGUE_LISTS.every((k) => Array.isArray((v as Record<string, unknown>)[k])) &&
  Array.isArray((v as SensitivityCatalogue).default_outcome_priority);

/** The sensitivity option catalogue served by the backend (no copy is kept here). */
export async function fetchLoopNullspaceSensitivityOptions(
  signal?: AbortSignal,
): Promise<SensitivityCatalogue> {
  const payload: unknown = await apiFetch<unknown>(`${LOOP_NULLSPACE_PATH}/sensitivity-options`, {
    signal,
  });
  if (!isCatalogue(payload)) {
    throw new Error('malformed catalogue from the loop-nullspace sensitivity-options endpoint');
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

// ------------------------------------------------- sensitivity panels (2.0.0)
export interface KernelHeatmap {
  outcome: string | null;
  units: string;
  shown: boolean;
  input_labels: string[];
  step_times_s: Array<number | null>;
  lower: Array<Array<number | null>>;
  upper: Array<Array<number | null>>;
  is_band: boolean;
  decision: DecisionView | null;
  reason: string;
}

export interface BudgetPart {
  input_class: string;
  variance: number | null;
  share: number | null;
  by_time: Array<number | null>;
}

export interface VarianceBudgetPanel {
  outcome: string | null;
  units: string;
  shown: boolean;
  step_times_s: Array<number | null>;
  parts: BudgetPart[];
  decision: DecisionView | null;
  reason: string;
}

export interface ParetoPoint {
  label: string;
  strategy: string;
  status: string;
  shown: boolean;
  rank: number | null;
  beats_baseline: boolean;
  variance_ratio: number | null;
  chs_mps: number | null;
  effort: number | null;
  effort_basis: string;
  mechanisms: string[];
  decision: DecisionView | null;
  reason: string;
}

export interface ParetoFront {
  name: string;
  labels: string[];
  withheld_count: number;
}

export interface SkippedArm {
  label: string;
  status: string;
  exclusion: string | null;
  reason: string;
}

export interface StrategyParetoPanel {
  mode: string | null;
  ranking_basis: string | null;
  points: ParetoPoint[];
  fronts: ParetoFront[];
  skipped: SkippedArm[];
  failure_rate: number | null;
  winner: string | null;
  assumptions: string[];
  non_claim: string | null;
  unavailable_reason: string | null;
}

export interface DispersionEntry {
  label: string;
  units: string;
  shown: boolean;
  lower: number | null;
  upper: number | null;
  is_band: boolean;
  decision: DecisionView | null;
  reason: string;
}

export interface McCheck {
  outcome: string;
  status: string;
  shown: boolean;
  predicted_sd: number | null;
  mc_sd: number | null;
  mc_sd_ci: Array<number | null>;
  gap: number | null;
  failure_rate: number | null;
  decision: DecisionView | null;
  reason: string;
}

export interface DispersionPanel {
  outcome: string | null;
  units: string;
  linear: DispersionEntry | null;
  arms: DispersionEntry[];
  mc: McCheck[];
  unavailable_reason: string | null;
}

export interface SensitivityPanels {
  status: string;
  reason: string | null;
  primary_outcome: string | null;
  skipped_priority: string[];
  outcomes: string[];
  declared_mechanisms: string[];
  noise_hypotheses: string[];
  linear_validity: string | null;
  cache_identity: string | null;
  non_claims: string[];
  kernel_heatmap: KernelHeatmap;
  variance_budget: VarianceBudgetPanel;
  strategy_pareto: StrategyParetoPanel;
  face_dispersion: DispersionPanel;
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
  /** Present in 2.0.0 models (NSX-99); absent in 1.0.0. */
  sensitivity?: SensitivityPanels;
}

const PANEL_KEYS = ['invariants', 'envelopes', 'hand_split', 'allocation', 'diagnostics'];

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/**
 * `value` as a plot model, or `null` when it is not one (fail closed).
 *
 * Accepted: 1.0.0 with its five panels, and 2.0.0 with the same five plus a
 * `sensitivity` record. Nothing is repaired or defaulted, so a model of any
 * other version, or a 2.0.0 one without its sensitivity panels, is refused
 * rather than drawn.
 */
export function asPlotModel(value: unknown): NSXPlotModel | null {
  if (!isRecord(value)) return null;
  const version = value.schema_version;
  if (version !== PLOT_MODEL_SCHEMA_VERSION && version !== PLOT_MODEL_SENSITIVITY_VERSION) {
    return null;
  }
  if (!PANEL_KEYS.every((k) => isRecord(value[k]))) return null;
  if (version === PLOT_MODEL_SENSITIVITY_VERSION && !isRecord(value.sensitivity)) return null;
  return value as unknown as NSXPlotModel;
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
  request: LoopNullspaceJobRequest,
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
