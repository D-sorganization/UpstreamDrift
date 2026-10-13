/**
 * View-model for the fail-closed golf-output policy payload (#10382, MS-108).
 *
 * Mirrors `output_policy_payload` in
 * src/shared/python/motion_matching/output_policy.py (the route returns it as
 * `output_policy`; it is an untyped dict so the generated API types do not
 * cover it). A row never carries a number unless the output is advertised.
 */

export const OUTPUT_POLICY_SCHEMA_VERSION = 'output-policy/1';

export const ADVERTISED_OUTPUTS = ['clubhead_position', 'clubhead_speed', 'sequencing'] as const;
export const WITHHELD_OUTPUTS = ['face_angle', 'joint_loads', 'impact', 'ball_flight'] as const;

export type OutputStatusKind = 'advertised' | 'validated' | 'withheld' | 'unavailable';

const STATUS_KINDS: readonly string[] = ['advertised', 'validated', 'withheld', 'unavailable'];
const UNAVAILABLE_LABEL = 'UNAVAILABLE';
const NO_PAYLOAD_REASON = 'policy status unavailable for this run';
const PERCENT_SCALE = 100;
const COVERAGE_DECIMALS = 1;

export interface OutputPolicyRow {
  name: string;
  status: OutputStatusKind;
  label: string;
  reason: string;
}

export interface FaceObservability {
  observable: boolean;
  reason: string;
  observed_frames: number;
  total_frames: number;
  coverage: number;
  gaps: Array<[number, number]>;
}

export interface OutputPolicyPayload {
  schema_version: string;
  outputs: OutputPolicyRow[];
  face_observability: FaceObservability | null;
}

export class OutputPolicySchemaError extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseRow(raw: unknown): OutputPolicyRow {
  if (
    !isRecord(raw) ||
    typeof raw.name !== 'string' ||
    typeof raw.status !== 'string' ||
    !STATUS_KINDS.includes(raw.status) ||
    typeof raw.label !== 'string' ||
    typeof raw.reason !== 'string'
  ) {
    throw new OutputPolicySchemaError('malformed output-policy row');
  }
  return {
    name: raw.name,
    status: raw.status as OutputStatusKind,
    label: raw.label,
    reason: raw.reason,
  };
}

function isGap(value: unknown): value is [number, number] {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    Number.isInteger(value[0]) &&
    Number.isInteger(value[1])
  );
}

function parseFace(raw: unknown): FaceObservability | null {
  if (raw === undefined || raw === null) return null;
  if (
    !isRecord(raw) ||
    typeof raw.observable !== 'boolean' ||
    typeof raw.reason !== 'string' ||
    !Number.isInteger(raw.observed_frames) ||
    !Number.isInteger(raw.total_frames) ||
    !Number.isFinite(raw.coverage) ||
    !Array.isArray(raw.gaps) ||
    !raw.gaps.every(isGap)
  ) {
    throw new OutputPolicySchemaError('malformed face_observability');
  }
  return raw as unknown as FaceObservability;
}

/** Parse the API payload; throws on anything but the one known schema (DbC). */
export function parseOutputPolicy(raw: unknown): OutputPolicyPayload {
  if (!isRecord(raw)) throw new OutputPolicySchemaError('output policy payload missing');
  if (raw.schema_version !== OUTPUT_POLICY_SCHEMA_VERSION) {
    throw new OutputPolicySchemaError(
      `unsupported output-policy schema_version: ${String(raw.schema_version)}`,
    );
  }
  if (!Array.isArray(raw.outputs)) throw new OutputPolicySchemaError('outputs must be a list');
  return {
    schema_version: raw.schema_version,
    outputs: raw.outputs.map(parseRow),
    face_observability: parseFace(raw.face_observability),
  };
}

/** Every known output unavailable: the fail-closed fallback. */
export function unavailableRows(reason: string = NO_PAYLOAD_REASON): OutputPolicyRow[] {
  return [...ADVERTISED_OUTPUTS, ...WITHHELD_OUTPUTS].map((name) => ({
    name,
    status: 'unavailable',
    label: UNAVAILABLE_LABEL,
    reason,
  }));
}

/** Formatted `value ± uncertainty unit`, only for an advertised output. */
export function advertisedValueText(
  row: OutputPolicyRow,
  receiptOutputs: unknown,
): string | null {
  if (row.status !== 'advertised' || !isRecord(receiptOutputs)) return null;
  const entry = receiptOutputs[row.name];
  if (!isRecord(entry)) return null;
  const { value, uncertainty, unit } = entry;
  if (!Number.isFinite(value) || !Number.isFinite(uncertainty)) return null;
  const suffix = typeof unit === 'string' && unit ? ` ${unit}` : '';
  return `${String(value)} ± ${String(uncertainty)}${suffix}`;
}

/** Coverage and gap runs of the face audit, or a not-audited notice. */
export function faceObservabilitySummary(face: FaceObservability | null): string {
  if (!face) return 'Face-angle observability: not audited for this receipt.';
  const gaps = face.gaps.map(([start, stop]) => `[${start}, ${stop})`).join(', ') || 'none';
  const verdict = face.observable ? 'observable' : 'unobservable';
  const coverage = (face.coverage * PERCENT_SCALE).toFixed(COVERAGE_DECIMALS);
  return (
    `Face-angle observability: ${verdict} (${face.reason}); observed ` +
    `${face.observed_frames}/${face.total_frames} frames (${coverage}%); unobserved runs: ${gaps}`
  );
}
