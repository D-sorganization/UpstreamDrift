/**
 * Provenance data shapes + pure helpers (epic #5968).
 *
 * Mirrors `src/shared/python/ux/provenance.py` so the React side shows
 * the same "why does this say 500?" description (DRY across platforms).
 * Kept separate from the `ProvenanceValue` component so the component
 * module only exports components (react-refresh).
 */

export interface ProvenanceRecord {
  formula: string;
  inputs: string[];
  source: string;
  computedAt: string;
  engine: string;
  runId: string;
}

export interface ProvenanceValueData {
  value: number | string;
  record: ProvenanceRecord;
  displayUnits?: string;
  label?: string;
}

/**
 * Build the multi-line provenance description.  Mirrors the Python
 * `ProvenanceValue.describe()` so both platforms show the same text.
 */
export function describeProvenance(pv: ProvenanceValueData): string {
  const unit = pv.displayUnits ? ` ${pv.displayUnits}` : '';
  const inputsLine =
    pv.record.inputs.length > 0
      ? `inputs: ${pv.record.inputs.join(', ')}`
      : '(no inputs)';
  return [
    `value: ${pv.value}${unit}`,
    `formula: ${pv.record.formula}`,
    inputsLine,
    `source: ${pv.record.source}`,
    `computed at: ${pv.record.computedAt}`,
  ].join('\n');
}

export function isLinked(pv: ProvenanceValueData): boolean {
  return pv.record.inputs.length > 0;
}
