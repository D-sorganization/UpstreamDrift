import { useId, useState } from "react";
import { displayTrace, type FlightRecord, type FlightTrace, type Point } from "./flightTrace";

// Shared by the flight polyline and its legend entry so the two never drift.
const FLIGHT_STROKE = "#60a5fa";

function Projection({
  trace,
  vertical,
  unit,
}: {
  trace: FlightTrace;
  vertical: 1 | 2;
  unit: "m" | "yd";
}) {
  const id = useId();
  const points = [...trace.flight_m, ...trace.rollout_m];
  // Both traces share bounds, including a backward or sideways rollout.
  let xmin = 0,
    xmax = 0,
    ymin = 0,
    ymax = 0;
  for (const p of points) {
    xmin = Math.min(xmin, p[0]);
    xmax = Math.max(xmax, p[0]);
    ymin = Math.min(ymin, p[vertical]);
    ymax = Math.max(ymax, p[vertical]);
  }
  const dx = Math.max(xmax - xmin, 1),
    dy = Math.max(ymax - ymin, 1);
  const x = (p: Point) => 48 + (280 * (p[0] - xmin)) / dx;
  const y = (p: Point) => 140 - (100 * (p[vertical] - ymin)) / dy;
  const line = (data: Point[]) => data.map((p) => `${x(p)},${y(p)}`).join(" ");
  const factor = unit === "yd" ? 1 / 0.9144 : 1;
  const label = (v: number) => (v * factor).toFixed(1);
  const title =
    vertical === 2 ? "Ball Flight: Downrange vs Height" : "Green Landing: Downrange vs Left";
  const landing = trace.flight_m[trace.flight_m.length - 1];
  const rest = trace.rollout_m[trace.rollout_m.length - 1];
  return (
    <svg
      viewBox="0 0 360 195"
      role="img"
      aria-labelledby={id}
      className="w-full min-w-0 rounded bg-canvas"
    >
      <title id={id}>{`${title} (${unit})`}</title>
      <g fill="#d1d5db" fontSize="11">
        <text x="12" y="18">
          {vertical === 2 ? "Ball Flight" : "Green Landing"}
        </text>
        <text x="12" y="33">{`${vertical === 2 ? "Height" : "Left"} (${unit})`}</text>
        <text x="40" y="44" textAnchor="end">
          {label(ymax)}
        </text>
        <text x="40" y="143" textAnchor="end">
          {label(ymin)}
        </text>
        <text x="48" y="158">
          {label(xmin)}
        </text>
        <text x="328" y="158" textAnchor="end">
          {label(xmax)}
        </text>
        <text x="188" y="178" textAnchor="middle">{`Downrange (${unit})`}</text>
      </g>
      <path d="M48 40 V140 H328" fill="none" stroke="#6b7280" />
      <polyline points={line(trace.flight_m)} fill="none" stroke={FLIGHT_STROKE} strokeWidth="2.5" />
      <polyline
        points={line(trace.rollout_m)}
        fill="none"
        stroke="#4ade80"
        strokeWidth="2.5"
        strokeDasharray="5 3"
      />
      <circle cx={x(landing)} cy={y(landing)} r="4" fill="#fbbf24">
        <title>Landing</title>
      </circle>
      <circle cx={x(rest)} cy={y(rest)} r="4" fill="#f472b6">
        <title>Rest</title>
      </circle>
    </svg>
  );
}

/** Consume existing model coordinates; absent/invalid/refused data never draws. */
export function BunkerFlightView({ record }: { record: FlightRecord }) {
  const [unit, setUnit] = useState<"m" | "yd">("m");
  const trace = displayTrace(record);
  if (!trace)
    return (
      <p role="status" className="mt-3 text-amber-200">
        {record.refused
          ? "REFUSED — no flight trace."
          : "Ball flight unavailable — no valid release trace."}
      </p>
    );
  const factor = unit === "yd" ? 1 / 0.9144 : 1;
  return (
    <section aria-label="Ball Flight and Green Landing" className="mt-4 min-w-0 space-y-2">
      <h4 className="font-medium text-white">Ball Flight and Green Landing</h4>
      <p className="text-xs text-amber-200">{trace.assumption}</p>
      <label className="flex items-center gap-2 text-xs text-token-text">
        Flight units
        <select
          value={unit}
          onChange={(e) => setUnit(e.target.value as "m" | "yd")}
          className="rounded bg-surface p-1"
        >
          <option value="m">Metres</option>
          <option value="yd">Yards</option>
        </select>
      </label>
      <p className="text-sm text-token-text">{`Carry ${(trace.carry_m * factor).toFixed(
        2,
      )} ${unit}`}</p>
      <div className="grid min-w-0 grid-cols-1 gap-2 xl:grid-cols-2">
        <Projection trace={trace} vertical={2} unit={unit} />
        <Projection trace={trace} vertical={1} unit={unit} />
      </div>
      <p className="flex flex-wrap gap-x-3 text-xs">
        <span style={{ color: FLIGHT_STROKE }}>Flight — solid</span>
        <span className="text-green-400">Rollout — dashed</span>
        <span className="text-amber-300">● Landing</span>
        <span className="text-pink-400">● Rest</span>
      </p>
    </section>
  );
}
