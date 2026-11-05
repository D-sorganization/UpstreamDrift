/**
 * Counter-torque explorer (#14470): sliders bounded by the core's ranges and a
 * summary read off the core's POST /counter-torque series (golden-pinned).
 */

import type { CounterTorqueRequest } from "@/api/generated/types";

type SliderKey = Exclude<keyof CounterTorqueRequest, "putter" | "n_points">;

const NM = " N·m";
const control = (
  key: SliderKey,
  label: string,
  unit: string,
  [min, max]: [number, number],
  step = 0.05,
) => ({ key, label, unit, min, max, step });

export type CounterTorqueControl = ReturnType<typeof control>;

/** Sliders, bounded by the core's PRELOAD/NOISE/GUST/LIE ranges. */
export const COUNTER_TORQUE_CONTROLS: readonly CounterTorqueControl[] = [
  control("preload_nm", "Grip Preload", NM, [0, 2]),
  control("off_preload_nm", "Off-Plane Preload", NM, [0, 2]),
  control("noise_scale", "Hand Noise", "", [0, 2]),
  control("gust_impulse_nm", "Gust Strength", NM, [0, 2]),
  control("lie_deg", "Shaft Lie", "°", [55, 85], 1),
];

/** The core's CounterTorqueInputs defaults and stroke_series grid. */
export const COUNTER_TORQUE_DEFAULTS: CounterTorqueRequest = {
  preload_nm: 0.6,
  off_preload_nm: 0.55,
  noise_scale: 0.25,
  gust_impulse_nm: 0.25,
  lie_deg: 70,
  putter: "toe_hang",
  n_points: 210,
};

export type CounterTorqueSamples = Record<string, number[]>;

function series(samples: CounterTorqueSamples, key: string): number[] {
  const values = samples[key];
  if (!values?.length) {
    throw new TypeError(`Counter-torque series is missing ${key}`);
  }
  return values;
}

const peak = (samples: CounterTorqueSamples, key: string) =>
  Math.max(...series(samples, key).map(Math.abs));

/** Impact face and path, peak in-plane hand forces, gust wobble peaks. */
export function ctSummary(samples: CounterTorqueSamples) {
  const last = (key: string) => {
    const values = series(samples, key);
    return values[values.length - 1];
  };
  return {
    faceDeg: last("face_angle_impact_deg"),
    pathDeg: last("club_path_impact_deg"),
    peakForceLeftN: peak(samples, "force_left_in_n"),
    peakForceRightN: peak(samples, "force_right_in_n"),
    gustPreloadDeg: peak(samples, "phi_preload_deg"),
    gustSlackDeg: peak(samples, "phi_slack_deg"),
  };
}

export type CounterTorqueSummary = ReturnType<typeof ctSummary>;

const signed = (deg: number) => `${deg >= 0 ? "+" : ""}${deg.toFixed(2)}°`;

/** The explorer's readout lines. */
export function ctReadout(s: CounterTorqueSummary): string[] {
  return [
    `Face at impact ${signed(s.faceDeg)}, club path ${signed(s.pathDeg)}`,
    `Peak in-plane hand force: left ${s.peakForceLeftN.toFixed(1)} N, right ${s.peakForceRightN.toFixed(1)} N`,
    `Gust wobble: ${s.gustPreloadDeg.toFixed(2)}° with grip preload vs ${s.gustSlackDeg.toFixed(2)}° slack`,
  ];
}
