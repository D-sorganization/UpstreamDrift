import type { SegmentForceFrame } from '../../api/segmentLoads';
export type { SegmentForceFrame } from '../../api/segmentLoads';

/** Fail closed for unavailable, ambiguous or stale load frames. */
export function segmentForcesAtTime(frame: SegmentForceFrame | undefined, time: number): Record<string, number | null> {
  if (!frame || !Number.isFinite(time) || frame.time_s !== time || frame.units !== 'N'
    || frame.sign_convention !== 'tension-positive' || typeof frame.source !== 'string'
    || !frame.source.trim() || !frame.values_n || typeof frame.values_n !== 'object'
    || Array.isArray(frame.values_n)) return {};
  return Object.fromEntries(Object.entries(frame.values_n).map(([key, value]) =>
    [key, typeof value === 'number' && Number.isFinite(value) ? value : null]));
}
