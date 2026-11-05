/**
 * Display units for the React frontends (#14458), mirroring
 * src/shared/python/ui/units.py: physics stays SI; imperial shows ft, mph and
 * inches (small offsets). units.golden.json pins the conversions to units.py.
 */

export type UnitSystem = "metric" | "imperial";
export type Quantity =
  "distance" | "speed" | "smallLength" | "breakLength" | "force" | "torque";

export const UNIT_SYSTEMS: readonly UnitSystem[] = ["metric", "imperial"];

const FT_TO_M = 0.3048;
const LBF_TO_N = 0.45359237 * 9.80665; // a pound under standard g, as units.py
const IN_LBF_TO_NM = (FT_TO_M / 12) * LBF_TO_N;
const STORAGE_KEY = "ud.unitSystem";

/** SI value of one display unit, and its suffix, per quantity and system. */
const UNITS: Record<Quantity, Record<UnitSystem, [number, string]>> = {
  distance: { metric: [1, " m"], imperial: [FT_TO_M, " ft"] },
  speed: { metric: [1, " m/s"], imperial: [0.44704, " mph"] },
  smallLength: { metric: [0.001, " mm"], imperial: [FT_TO_M / 12, " in"] },
  // Putt break as format_read shows it: cm, not mm, in metric (#14466).
  breakLength: { metric: [0.01, " cm"], imperial: [FT_TO_M / 12, " in"] },
  force: { metric: [1, " N"], imperial: [LBF_TO_N, " lbf"] },
  torque: { metric: [1, " N·m"], imperial: [IN_LBF_TO_NM, " in·lbf"] },
};

export function toDisplay(si: number, q: Quantity, s: UnitSystem): number {
  return si / UNITS[q][s][0];
}

export function fromDisplay(value: number, q: Quantity, s: UnitSystem): number {
  return value * UNITS[q][s][0];
}

export function unitSuffix(q: Quantity, s: UnitSystem): string {
  return UNITS[q][s][1];
}

export function formatQuantity(
  si: number,
  q: Quantity,
  s: UnitSystem,
  decimals: number,
): string {
  return `${toDisplay(si, q, s).toFixed(decimals)}${unitSuffix(q, s)}`;
}

function isUnitSystem(value: unknown): value is UnitSystem {
  return value === "metric" || value === "imperial";
}

/** Stored preference, falling back to metric like units.py. */
export function loadUnitSystem(): UnitSystem {
  try {
    const stored = globalThis.localStorage?.getItem(STORAGE_KEY);
    return isUnitSystem(stored) ? stored : "metric";
  } catch {
    return "metric";
  }
}

export function saveUnitSystem(system: UnitSystem): void {
  if (!isUnitSystem(system)) {
    throw new TypeError(`Unknown unit system: ${String(system)}`);
  }
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, system);
  } catch {
    // Storage can be unavailable (private mode); the choice still applies.
  }
}
