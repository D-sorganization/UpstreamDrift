/**
 * Pure helpers over the generated FieldMetadata registry (epic #5968).
 *
 * Kept separate from the `HelpfulField` component so the component
 * module only exports components (react-refresh), and separate from the
 * auto-generated `fieldMetadata.ts` so regeneration never clobbers them.
 */

import type { FieldMetadata } from './fieldMetadata';

export function isNumericRange(
  range: FieldMetadata['validRange'],
): range is [number, number] {
  return (
    Array.isArray(range) &&
    range.length === 2 &&
    typeof range[0] === 'number' &&
    typeof range[1] === 'number'
  );
}

export function isInRange(meta: FieldMetadata, value: number): boolean {
  if (!isNumericRange(meta.validRange)) {
    return true;
  }
  const [lo, hi] = meta.validRange;
  return value >= lo && value <= hi;
}
