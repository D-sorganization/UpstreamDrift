/** Producer-owned axial loads at declared segment sections. */
export interface SegmentForceFrame {
  time_s: number;
  source: string;
  units: 'N';
  sign_convention: 'tension-positive';
  values_n: Record<string, number | null>;
}

/** Fail closed for unavailable, ambiguous or stale load frames. */
export function segmentForcesAtTime(frame: SegmentForceFrame | undefined, time: number): Record<string, number | null> {
  if (!frame || !Number.isFinite(time) || frame.time_s !== time || frame.units !== 'N'
    || frame.sign_convention !== 'tension-positive' || typeof frame.source !== 'string'
    || !frame.source.trim() || !frame.values_n || typeof frame.values_n !== 'object'
    || Array.isArray(frame.values_n)) return {};
  return Object.fromEntries(Object.entries(frame.values_n).map(([key, value]) =>
    [key, typeof value === 'number' && Number.isFinite(value) ? value : null]));
}
