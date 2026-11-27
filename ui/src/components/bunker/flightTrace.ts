/** Display-only guard for the additive #15555 payload; no flight physics. */
export type Point = [number, number, number];
export interface FlightTrace {
  flight_m: Point[];
  rollout_m: Point[];
  carry_m: number;
  assumption: string;
}
export interface FlightRecord {
  refused: boolean;
  carry: { carry_m: number } | null;
  flight?: unknown;
}

function path(value: unknown): value is Point[] {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    value.every(
      (p) =>
        Array.isArray(p) &&
        p.length === 3 &&
        p.every((v) => typeof v === "number" && Number.isFinite(v)),
    )
  );
}

/** Return only a complete finite trace paired with its matching carry. */
export function displayTrace(record: FlightRecord): FlightTrace | null {
  if (
    record.refused ||
    record.carry === null ||
    !Number.isFinite(record.carry.carry_m) ||
    !record.flight ||
    typeof record.flight !== "object"
  )
    return null;
  const f = record.flight as FlightTrace;
  if (
    !path(f.flight_m) ||
    !path(f.rollout_m) ||
    !Number.isFinite(f.carry_m) ||
    f.carry_m < 0 ||
    typeof f.assumption !== "string" ||
    !f.assumption.trim()
  )
    return null;
  const landing = f.flight_m[f.flight_m.length - 1];
  const close = (a: number, b: number) =>
    Math.abs(a - b) <= 1e-7 * Math.max(1, Math.abs(a), Math.abs(b));
  if (
    !close(f.carry_m, record.carry.carry_m) ||
    !close(f.carry_m, Math.hypot(landing[0], landing[1])) ||
    !close(landing[2], 0) ||
    !landing.every((v, i) => close(v, f.rollout_m[0][i]))
  )
    return null;
  return f;
}
