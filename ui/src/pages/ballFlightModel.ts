import { getFieldMetadata } from '@/ux/fieldMetadata';
import { isInRange } from '@/ux/fieldHelpers';

/** Mirrors the desktop tracer's per-model color cycle. */
export const MODEL_COLORS = [
  '#ef4444', // red
  '#3b82f6', // blue
  '#22c55e', // green
  '#e879f9', // magenta
  '#eab308', // yellow
  '#06b6d4', // cyan
  '#f97316', // orange
] as const;

/** Launch-condition field ids (registry-backed; single source of units). */
export const LAUNCH_FIELD_IDS = [
  'ball_flight.ball_speed',
  'ball_flight.launch_angle',
  'ball_flight.azimuth_angle',
  'ball_flight.spin_rate',
  'ball_flight.spin_axis_tilt',
  'ball_flight.wind_speed',
  'ball_flight.wind_direction',
] as const;

export type LaunchFieldId = (typeof LAUNCH_FIELD_IDS)[number];

/**
 * Validate the form values against the metadata registry.
 *
 * Returns the field ids whose values are missing, non-numeric, or outside
 * the declared valid range.
 */
export function invalidLaunchFields(
  values: Record<LaunchFieldId, string>,
): LaunchFieldId[] {
  return LAUNCH_FIELD_IDS.filter((fieldId) => {
    const raw = values[fieldId];
    const parsed = Number(raw);
    if (raw.trim() === '' || Number.isNaN(parsed)) return true;
    return !isInRange(getFieldMetadata(fieldId), parsed);
  });
}

/** Color for the i-th selected model (cycles like the desktop palette). */
export function modelColor(index: number): string {
  return MODEL_COLORS[index % MODEL_COLORS.length];
}
