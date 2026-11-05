/**
 * Saved scenarios for the React putting page (#14472) over the shared core's
 * /scenario and /scenario-compare routes (service_setup, #14471).
 * puttingSetup.golden.json pins the setup mapping to the core.
 */

import type { Putt3DSimulationRequest } from "@/api/generated/types";
import { dispersionArgs, type Noise } from "./dispersion";

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
