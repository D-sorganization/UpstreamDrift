import type {
  AimSolve3DResponse,
  BalancePresetsResponse,
  CounterTorqueRequest,
  CounterTorqueResponse,
  Putt3DSimulationRequest,
  Putt3DSimulationResponse,
} from "./generated/types";
import type { DispersionArgs } from "@/utils/dispersion";
import type { GreenPresets } from "@/utils/greenTypes";
import type { Comparison, ImportedSetup, Scenario } from "@/utils/puttingSetup";
import type {
  StrokesGainedArgs,
  StrokesGainedReport,
} from "@/utils/strokesGained";

import { apiFetch } from "./fetch";

/** Execute the canonical Python model; the browser never mirrors its laws. */
export function simulatePutt3D(
  request: Putt3DSimulationRequest,
): Promise<Putt3DSimulationResponse> {
  return apiFetch<Putt3DSimulationResponse>(
    "/api/tools/putting-green/simulate-3d",
    {
      method: "POST",
      body: JSON.stringify(request),
    },
  );
}

/** The core's holing pace and line for this green (solve_aim, #14466). */
export function solveAim3D(
  request: Putt3DSimulationRequest,
): Promise<AimSolve3DResponse> {
  return apiFetch<AimSolve3DResponse>("/api/tools/putting-green/solve-aim-3d", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/** The shared core's named putter heads (putter_presets.BALANCE_PRESETS). */
export function fetchPutterPresets(): Promise<BalancePresetsResponse> {
  return apiFetch<BalancePresetsResponse>(
    "/api/tools/putting-green/putter-presets",
  );
}

/** Hand couples, forces and gust response over the stroke (#14469). */
export function fetchCounterTorque(
  request: CounterTorqueRequest,
): Promise<CounterTorqueResponse> {
  return apiFetch<CounterTorqueResponse>(
    "/api/tools/putting-green/counter-torque",
    { method: "POST", body: JSON.stringify(request) },
  );
}

/** The shared core's named green shapes, by preset key (#14464). */
export function fetchGreenPresets(): Promise<GreenPresets> {
  return apiFetch<GreenPresets>("/api/tools/putting-green/green-presets");
}

/** Rest positions of a noisy batch (core service_dispersion.dispersion_study). */
export interface DispersionReport {
  n: number;
  distance_ft: number;
  /** Past the hole is positive. */
  rest_x_ft: number[];
  /** Left of the line is positive. */
  rest_y_ft: number[];
  holed: boolean[];
  make_rate: number;
  mean_leave_ft: number | null;
}

/** Make rate by distance (core service_dispersion.make_spread_study). */
export interface MakeSpreadReport {
  distances_ft: number[];
  make_rate: number[];
  samples_per_distance: number;
}

/**
 * Core studies roll whole batches: the make-% spread plans one line fan per
 * distance on a sloped green and can take a minute or more, which the desktop
 * runs on a worker. The 15 s default would abort them.
 */
export const STUDY_TIMEOUT_MS = 180_000;

function postAnalytics<T>(route: string, args: object): Promise<T> {
  return apiFetch<T>(`/api/tools/putting-analytics/${route}`, {
    method: "POST",
    body: JSON.stringify(args),
    timeoutMs: STUDY_TIMEOUT_MS,
  });
}

export function runDispersion(args: DispersionArgs): Promise<DispersionReport> {
  return postAnalytics<DispersionReport>("dispersion", args);
}

export function runMakeSpread(args: DispersionArgs): Promise<MakeSpreadReport> {
  return postAnalytics<MakeSpreadReport>("make-spread", args);
}

/** The player's E(d) and the putt's strokes gained (core service_sg.sg_study). */
export function runStrokesGained(
  args: StrokesGainedArgs,
): Promise<StrokesGainedReport> {
  return postAnalytics<StrokesGainedReport>("strokes-gained", args);
}

/** A named setup and its dispersion result (core service_setup, #14471). */
export function runScenario(args: object): Promise<Scenario> {
  return postAnalytics<Scenario>("scenario", args);
}

/** Each scenario's make rate and leave against the first. */
export function compareScenarios(args: {
  scenarios: Scenario[];
}): Promise<Comparison> {
  return postAnalytics<Comparison>("scenario-compare", args);
}

/** The versioned setup document (schema_version, suite, setup, scenarios). */
export function exportSetup(args: object): Promise<object> {
  return postAnalytics<object>("setup-export", args);
}

export function importSetup(args: {
  document: unknown;
}): Promise<ImportedSetup> {
  return postAnalytics<ImportedSetup>("setup-import", args);
}
