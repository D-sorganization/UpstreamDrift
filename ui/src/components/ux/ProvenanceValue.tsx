/**
 * ProvenanceValue — value display carrying its derivation (epic #5968).
 *
 * Parity with the PyQt6 `ProvenanceValueLabel`: renders a value (plus
 * display units) and exposes a provenance description via
 * `aria-describedby` + `title` so a hover or screen-reader answers
 * "why does this say 500?".  Values derived from named inputs render a
 * "linked" badge affordance; the badge's exact visual styling is a
 * human design decision (deferred).
 *
 * Data shapes + pure helpers live in `../../ux/provenance` (DRY).
 */

import { useId } from 'react';
import {
  describeProvenance,
  isLinked,
  type ProvenanceValueData,
} from '../../ux/provenance';

interface ProvenanceValueProps {
  data: ProvenanceValueData;
}

export function ProvenanceValue({ data }: ProvenanceValueProps) {
  const descId = useId();
  const description = describeProvenance(data);
  const linked = isLinked(data);
  const unit = data.displayUnits ? ` ${data.displayUnits}` : '';

  return (
    <span className="inline-flex items-center gap-1">
      <span
        aria-describedby={descId}
        title={description}
        aria-label={data.label}
        className="font-mono text-gray-200"
      >
        {data.value}
        {unit}
      </span>
      {linked && (
        <span
          data-testid="provenance-link-badge"
          aria-label="derived value — has provenance link"
          className="text-xs text-blue-400"
        >
          🔗
        </span>
      )}
      <span id={descId} className="sr-only">
        {description}
      </span>
    </span>
  );
}
