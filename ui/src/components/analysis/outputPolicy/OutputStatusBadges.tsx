import { Badge, type BadgeTone } from '@/components/ui';
import {
  OutputPolicySchemaError,
  advertisedValueText,
  faceObservabilitySummary,
  parseOutputPolicy,
  unavailableRows,
  type FaceObservability,
  type OutputPolicyRow,
  type OutputStatusKind,
} from './outputPolicy';

const TONES: Record<OutputStatusKind, BadgeTone> = {
  advertised: 'success',
  validated: 'success',
  withheld: 'warning',
  unavailable: 'neutral',
};

export interface OutputStatusBadgesProps {
  /** `output_policy` from the matched-swing receipt route. */
  payload: unknown;
  /** The route's `receipt`; read only for advertised value and uncertainty. */
  receipt?: unknown;
}

function resolve(payload: unknown): {
  rows: OutputPolicyRow[];
  face: FaceObservability | null;
  error: string | null;
} {
  if (payload === undefined || payload === null) {
    return { rows: unavailableRows(), face: null, error: null };
  }
  try {
    const parsed = parseOutputPolicy(payload);
    return { rows: parsed.outputs, face: parsed.face_observability, error: null };
  } catch (err: unknown) {
    const message = err instanceof OutputPolicySchemaError ? err.message : 'invalid output policy';
    return { rows: unavailableRows(message), face: null, error: message };
  }
}

/** Per-output policy badges; withheld and unavailable outputs show no number. */
export function OutputStatusBadges({ payload, receipt }: OutputStatusBadgesProps) {
  const { rows, face, error } = resolve(payload);
  const receiptOutputs = (receipt as { outputs?: unknown } | undefined)?.outputs;
  return (
    <section aria-label="Output status" className="space-y-1 text-xs">
      {error && (
        <p role="alert" className="text-red-300">
          {error}
        </p>
      )}
      <ul className="space-y-1">
        {rows.map((row) => {
          const value = advertisedValueText(row, receiptOutputs);
          return (
            <li key={row.name} data-testid={`output-row-${row.name}`} className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-gray-300">{row.name}</span>
              <Badge tone={TONES[row.status]} data-testid={`output-status-${row.name}`}>
                {row.label.toUpperCase()}
              </Badge>
              {value && (
                <span data-testid={`output-value-${row.name}`} className="text-gray-100">
                  {value}
                </span>
              )}
              <span className="text-gray-400">{row.reason}</span>
            </li>
          );
        })}
      </ul>
      <p className="text-gray-400">{faceObservabilitySummary(face)}</p>
    </section>
  );
}
