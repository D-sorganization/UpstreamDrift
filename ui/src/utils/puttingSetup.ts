/**
 * Saved scenarios and JSON setup files for the React putting page (#14472)
 * over the shared core's /scenario, /scenario-compare, /setup-export and
 * /setup-import routes (service_setup, #14471). puttingSetup.golden.json pins
 * the setup mapping to the core.
 */

import type { Putt3DSimulationRequest } from "@/api/generated/types";
import { dispersionArgs, type Noise } from "./dispersion";
import { FT_TO_M } from "./units";

/** The core's setup document version (service_setup.SCHEMA_VERSION). */
export const SCHEMA_VERSION = "4.0.0";
export const MAX_SCENARIOS = 50;
/** Putts rolled per saved scenario. */
export const SCENARIO_N = 100;

/** A core setup (service_setup.SETUP_KEYS), in feet, percent and degrees. */
export type Setup = Record<string, number>;

export interface Scenario {
  name: string;
  setup: Setup;
  result: { make_rate: number; mean_leave_ft: number | null };
}

/** Each scenario against the first (service_setup.compare_scenarios). */
export interface Comparison {
  baseline: string;
  rows: { make_rate_delta: number; mean_leave_delta_ft: number | null }[];
}

/** Studio scenario rows are formatted text; the core only counts them. */
export interface ImportedSetup {
  setup: Setup;
  scenarios: Scenario[];
  studio_scenarios: number;
}

/** Batch-only arguments that are not part of a setup. */
const NOT_SETUP = ["n", "seed", "strike_toe_sd_mm"];

/** The page's putt and spread as core setup arguments; the core fills the rest. */
export function setupArgs(
  parameters: Putt3DSimulationRequest,
  noise: Noise,
): Setup {
  const args = Object.entries(dispersionArgs(parameters, noise, 1));
  return Object.fromEntries(args.filter(([key]) => !NOT_SETUP.includes(key)));
}

/**
 * Page parameter changes and spread for a core setup. The hole moves to the
 * setup's distance along the page's current line, and the fall line keeps
 * its angle to that line, so setupArgs of the result returns the setup.
 */
export function applySetup(
  parameters: Putt3DSimulationRequest,
  noise: Noise,
  setup: Setup,
): { parameters: Partial<Putt3DSimulationRequest>; noise: Noise } {
  const line = Math.atan2(parameters.hole_y_m, parameters.hole_x_m);
  const distanceM = setup.distance_ft * FT_TO_M;
  const spread = Object.fromEntries(
    Object.keys(noise).map((key) => [
      key,
      setup[key] ?? noise[key as keyof Noise],
    ]),
  ) as Noise;
  return {
    parameters: {
      hole_x_m: distanceM * Math.cos(line),
      hole_y_m: distanceM * Math.sin(line),
      grade_percent: setup.grade_pct,
      downhill_aspect_deg: setup.aspect_deg + (line * 180) / Math.PI,
      stimp_rating: setup.stimp,
    },
    noise: spread,
  };
}

export function setupFileName(): string {
  return `putting-setup-v${SCHEMA_VERSION}.json`;
}
