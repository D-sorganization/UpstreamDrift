/**
 * Auto-Solve read for the React frontends (#14466), mirroring the desktop
 * putting_green_gui format_read. The pace and line come from the core's
 * /solve-aim-3d; autoSolve.golden.json pins this text to the desktop's.
 */

import type { AimSolve3DResponse } from "@/api/generated/types";
import { formatQuantity, type UnitSystem } from "./units";

export type SolvedControls = Pick<
  AimSolve3DResponse,
  "putter_speed_mps" | "aim_heading_deg"
>;

/** The request fields the solve fills in, as the desktop writes its spins. */
export function solvedControls(solve: AimSolve3DResponse): SolvedControls {
  return {
    putter_speed_mps: solve.putter_speed_mps,
    aim_heading_deg: solve.aim_heading_deg,
  };
}

/** An aim heading as the desktop prints it (`{:+.1f}°`). */
export function signedDegrees(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}°`;
}

/** One line naming the solved read; break_m >= 0 aims left of the cup. */
export function formatRead(
  solve: AimSolve3DResponse,
  system: UnitSystem,
): string {
  const metric = system === "metric";
  const speed = formatQuantity(
    solve.putter_speed_mps,
    "speed",
    system,
    metric ? 2 : 1,
  );
  const lean = formatQuantity(
    Math.abs(solve.break_m),
    "breakLength",
    system,
    1,
  );
  const side = solve.break_m >= 0 ? "left" : "right";
  const found = solve.holed ? "" : " (no holing line found)";
  return (
    `Optimum read:  ${speed} at ${signedDegrees(solve.aim_heading_deg)}, ` +
    `aim ${lean} ${side} of the cup${found}`
  );
}
