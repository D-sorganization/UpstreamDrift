import type { Putt3DSampleResponse } from "@/api/generated/types";

/** Slow-motion approach time shown before the physics trajectory starts. */
export const IMPACT_LEAD_IN_S = 0.18;

function interpolate(
  left: Putt3DSampleResponse,
  right: Putt3DSampleResponse,
  alpha: number,
): Putt3DSampleResponse {
  const mix = (a: number, b: number) => a + (b - a) * alpha;
  return {
    t_s: mix(left.t_s, right.t_s),
    x_m: mix(left.x_m, right.x_m),
    y_m: mix(left.y_m, right.y_m),
    z_m: mix(left.z_m, right.z_m),
    speed_mps: mix(left.speed_mps, right.speed_mps),
    spin_rad_s: mix(left.spin_rad_s, right.spin_rad_s),
    mode: alpha + 1e-12 < 0.5 ? left.mode : right.mode,
  };
}

/** Resolve a playback time to a smooth, clamped physical sample. */
export function sampleAtPlaybackTime(
  samples: Putt3DSampleResponse[],
  playbackTimeS: number,
): Putt3DSampleResponse {
  if (samples.length === 0) {
    throw new Error("Putting playback requires at least one trajectory sample");
  }
  const physicsTimeS = Math.max(0, playbackTimeS - IMPACT_LEAD_IN_S);
  if (physicsTimeS <= samples[0].t_s) return samples[0];
  const last = samples[samples.length - 1];
  if (physicsTimeS >= last.t_s) return last;

  let low = 0;
  let high = samples.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (samples[middle].t_s <= physicsTimeS) low = middle;
    else high = middle;
  }
  const left = samples[low];
  const right = samples[high];
  const interval = right.t_s - left.t_s;
  const alpha = interval > 0 ? (physicsTimeS - left.t_s) / interval : 0;
  return interpolate(left, right, alpha);
}

/** Marker rotation exposes signed skid/roll spin instead of hiding it. */
export function ballRotationRad(
  sample: Putt3DSampleResponse,
  physicsTimeS: number,
): number {
  return sample.spin_rad_s * Math.max(0, physicsTimeS);
}
