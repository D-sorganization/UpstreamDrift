/**
 * Noise channels and dispersion-batch arguments for the React putting page
 * (#14460), mirroring the desktop batch panel (_batch_panel.py) over the
 * shared core's /dispersion and /make-spread routes. dispersion.golden.json
 * pins the channels, batch sizes and argument mapping to the core.
 */

import type { Putt3DSimulationRequest } from "@/api/generated/types";

export type NoiseKey =
  | "face_sd_deg"
  | "path_sd_deg"
  | "speed_cv"
  | "strike_toe_sd_mm"
  | "grade_rel_sd"
  | "aspect_sd_deg"
  | "stimp_rel_sd";

export type Noise = Record<NoiseKey, number>;

export interface NoiseChannel {
  key: NoiseKey;
  label: string;
  /** Core default and bounds (service_variance._STROKE/_READ_BOUNDS). */
  initial: number;
  min: number;
  max: number;
  /** Core units per shown unit (fractions are shown in percent). */
  perShown: number;
  suffix: string;
}

const channel = (
  key: NoiseKey,
  label: string,
  [initial, min, max]: [number, number, number],
  perShown: number,
  suffix: string,
): NoiseChannel => ({ key, label, initial, min, max, perShown, suffix });

export const NOISE_CHANNELS: readonly NoiseChannel[] = [
  channel("face_sd_deg", "Face SD", [1.5, 0, 10], 1, "°"),
  channel("path_sd_deg", "Path SD", [2, 0, 15], 1, "°"),
  channel("speed_cv", "Speed CV", [0.08, 0, 0.5], 0.01, "%"),
  channel("strike_toe_sd_mm", "Strike SD", [0, 0, 25], 1, "mm"),
  channel("grade_rel_sd", "Read Slope SD", [0, 0, 1], 0.01, "%"),
  channel("aspect_sd_deg", "Read Fall Line SD", [0, 0, 45], 1, "°"),
  channel("stimp_rel_sd", "Read Speed SD", [0, 0, 0.5], 0.01, "%"),
];

export const BATCH_SIZES: readonly number[] = [40, 100, 1000];
/** Putts per distance for the make-% spread, as on the desktop. */
export const SPREAD_N = 60;

const FT_TO_M = 0.3048;

export function defaultNoise(): Noise {
  return Object.fromEntries(
    NOISE_CHANNELS.map((c) => [c.key, c.initial]),
  ) as Noise;
}

/** Wrap an angle to [-180, 180) degrees. */
function wrapDeg(angle: number): number {
  return ((((angle + 180) % 360) + 360) % 360) - 180;
}

/**
 * Core arguments for the simulator's putt: distance along the line to the
 * hole, and the fall direction measured CCW from that line (both the core and
 * the 3D simulator build the plane with HeightField.planar). Assumes the
 * simulator's ball starts at the origin, so the hole position is the line.
 */
export function dispersionArgs(
  parameters: Putt3DSimulationRequest,
  noise: Noise,
  n: number,
) {
  if (!Number.isInteger(n) || n < 1) {
    throw new RangeError(`batch size must be a positive integer, got ${n}`);
  }
  const { hole_x_m, hole_y_m, grade_percent } = parameters;
  const lineDeg = (Math.atan2(hole_y_m, hole_x_m) * 180) / Math.PI;
  return {
    distance_ft: Math.hypot(hole_x_m, hole_y_m) / FT_TO_M,
    grade_pct: grade_percent,
    aspect_deg: grade_percent
      ? wrapDeg(parameters.downhill_aspect_deg - lineDeg)
      : 0,
    stimp: parameters.stimp_rating,
    seed: parameters.random_seed,
    ...noise,
    n,
  };
}

export type DispersionArgs = ReturnType<typeof dispersionArgs>;
