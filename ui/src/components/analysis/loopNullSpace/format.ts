import type { DisplayClass } from '@/api/loopNullSpace';

/** Display-only formatting for plot-model numbers (no model quantity is derived). */

/** A number for display; `null` (unavailable) is the word, never a zero. */
export function fmt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'unavailable';
  }
  return String(Number(value.toPrecision(5)));
}

/** A list of numbers for display, `unavailable` entries kept as such. */
export function fmtList(values: ReadonlyArray<number | null>): string {
  return values.map(fmt).join(', ');
}

/** Visible words for a segment's display class (text, not colour, carries it). */
export const DISPLAY_CLASS_TEXT: Record<DisplayClass, string> = {
  DETERMINED_SIGN: 'Determined sign',
  INDETERMINATE: 'Indeterminate (shaded)',
  UNBOUNDED: 'Unbounded (hatched)',
};

/** Frame indices from comma-separated text; `null` when any entry is not a whole number. */
export function parseFrameIndices(text: string): number[] | null {
  const parts = text.split(',').map((p) => p.trim());
  if (parts.some((p) => !/^\d+$/.test(p))) return null;
  return parts.map(Number);
}
