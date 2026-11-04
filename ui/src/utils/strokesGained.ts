/** Strokes-gained arguments and readout (#14468), as the desktop _sg_panel.py;
 * strokesGained.golden.json pins them to the core's /strokes-gained route. */

import type {
  Putt3DSimulationRequest,
  Putt3DSimulationResponse,
} from "@/api/generated/types";
import { dispersionArgs, type Noise } from "./dispersion";
import { formatQuantity, type UnitSystem } from "./units";

const FT_TO_M = 0.3048;

/** service_sg.DEFAULT_SG_N / DEFAULT_SG_PACES_FT and aggregate.DEFAULT_GRID_FT. */
export const SG_STUDY = {
  n: 200,
  paces_ft: [0.5, 1.5, 3.0],
  distances_ft: [0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 45],
};

/** The core's E(d) curve and the putt's strokes gained (service_sg.sg_study). */
export interface StrokesGainedReport {
  distance_ft: number;
  curve: { distances_ft: number[]; expected_putts: number[] };
  expected_putts: number;
  sg_player: number | null;
}

/** Distance from the simulated ball's rest point to the hole, in feet. */
export function puttLeaveFt(result: Putt3DSimulationResponse): number {
  if (result.holed) return 0;
  const rest = result.samples[result.samples.length - 1];
  const { hole_x_m, hole_y_m } = result.surface;
  return Math.hypot(rest.x_m - hole_x_m, rest.y_m - hole_y_m) / FT_TO_M;
}

/** The green and player of the batch panel, the study size and the putt. */
export function sgArgs(
  parameters: Putt3DSimulationRequest,
  noise: Noise,
  putt: { holed: boolean; leave_ft: number },
) {
  return {
    ...dispersionArgs(parameters, noise, SG_STUDY.n),
    ...SG_STUDY,
    holed: putt.holed,
    leave_ft: putt.holed ? 0 : putt.leave_ft,
  };
}

export type StrokesGainedArgs = ReturnType<typeof sgArgs>;

/** The desktop panel's summary line, in the user's units. */
export function sgSummary(
  report: StrokesGainedReport,
  system: UnitSystem,
): string {
  const start = formatQuantity(
    report.distance_ft * FT_TO_M,
    "distance",
    system,
    1,
  );
  const sg =
    report.sg_player === null
      ? ""
      : ` This putt gained ${report.sg_player >= 0 ? "+" : ""}${report.sg_player.toFixed(2)} strokes.`;
  return `From ${start} you average ${report.expected_putts.toFixed(2)} putts.${sg}`;
}
