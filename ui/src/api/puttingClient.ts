import type {
  Putt3DSimulationRequest,
  Putt3DSimulationResponse,
} from "./generated/types";
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
