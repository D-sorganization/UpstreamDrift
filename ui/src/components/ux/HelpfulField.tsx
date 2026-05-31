/**
 * HelpfulField — metadata-driven input wrapper (epic #5968, Phase 2).
 *
 * Consumes the generated `FIELD_METADATA` registry (derived from
 * `configs/ux/field_metadata.yaml`, the single source of truth — DRY).
 * It renders a labelled numeric/enum input wired up with
 * `aria-describedby` help text and surfaces an `onViolation` callback
 * when a numeric value breaches the field's declared range.
 *
 * Pure helpers live in `../../ux/fieldHelpers` (DRY / react-refresh).
 * The popover look / link-badge styling is a human design decision and
 * is intentionally NOT implemented here (deferred).
 */

import { useId } from 'react';
import { getFieldMetadata } from '../../ux/fieldMetadata';
import { isInRange, isNumericRange } from '../../ux/fieldHelpers';

interface HelpfulFieldProps {
  /** Dotted id present in the generated registry. */
  fieldId: string;
  /** Controlled value. */
  value: number | string;
  /** Change handler; receives the raw input string. */
  onChange: (value: string) => void;
  /** Fired when a numeric value breaches the declared range. */
  onViolation?: (fieldId: string, value: number) => void;
  disabled?: boolean;
}

export function HelpfulField({
  fieldId,
  value,
  onChange,
  onViolation,
  disabled,
}: HelpfulFieldProps) {
  // Throws for unknown ids — fail fast at the boundary (DbC).
  const meta = getFieldMetadata(fieldId);
  const helpId = useId();
  const inputId = useId();
  const numeric = isNumericRange(meta.validRange);
  const enumValues =
    Array.isArray(meta.validRange) && !numeric
      ? (meta.validRange as string[])
      : null;

  const handleChange = (raw: string) => {
    onChange(raw);
    if (numeric && onViolation) {
      const parsed = Number(raw);
      if (!Number.isNaN(parsed) && !isInRange(meta, parsed)) {
        onViolation(fieldId, parsed);
      }
    }
  };

  const unitSuffix = meta.units ? ` (${meta.units})` : '';

  return (
    <div className="space-y-1">
      <label
        htmlFor={inputId}
        className="block text-sm font-medium text-gray-300"
      >
        {meta.label}
        {unitSuffix}
      </label>
      {enumValues ? (
        <select
          id={inputId}
          value={String(value)}
          disabled={disabled}
          aria-describedby={helpId}
          onChange={(e) => handleChange(e.target.value)}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
        >
          {enumValues.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={inputId}
          type={numeric ? 'number' : 'text'}
          value={value}
          disabled={disabled}
          min={numeric ? meta.validRange![0] : undefined}
          max={numeric ? meta.validRange![1] : undefined}
          aria-describedby={helpId}
          onChange={(e) => handleChange(e.target.value)}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
        />
      )}
      <p id={helpId} className="text-xs text-gray-500" title={meta.defaultSource}>
        {meta.shortHelp}
      </p>
    </div>
  );
}
