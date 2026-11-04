import type {
  Putt3DSimulationRequest,
  Putt3DSimulationResponse,
} from "./generated/types";
import type { DispersionArgs } from "@/utils/dispersion";

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

function postAnalytics<T>(route: string, args: DispersionArgs): Promise<T> {
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
