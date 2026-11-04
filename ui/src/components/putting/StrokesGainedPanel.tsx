/** The player's expected putts and the simulated putt's strokes gained (#14468). */
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { runStrokesGained } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import {
  puttLeaveFt,
  sgArgs,
  sgSummary,
  type StrokesGainedReport,
} from "@/utils/strokesGained";
import {
  fromDisplay,
  toDisplay,
  unitSuffix,
  type UnitSystem,
} from "@/utils/units";

const FT_TO_M = 0.3048;
const W = 260;
const H = 160;
const PAD = 28;

const shownFt = (feet: number, system: UnitSystem) =>
  Number(toDisplay(feet * FT_TO_M, "distance", system).toFixed(2));

function CurveChart({
  report,
  system,
}: {
  report: StrokesGainedReport;
  system: UnitSystem;
}) {
  const { distances_ft, expected_putts } = report.curve;
  const maxD = Math.max(...distances_ft);
  const maxE = Math.max(2, Math.ceil(Math.max(...expected_putts)));
  const x = (d: number) => PAD + (d / maxD) * (W - 1.5 * PAD);
  const y = (e: number) => H - PAD - ((e - 1) / (maxE - 1)) * (H - 1.5 * PAD);
  const points = distances_ft.map((d, i) => [x(d), y(expected_putts[i])]);
  const unit = unitSuffix("distance", system).trim();
  return (
    <svg
      role="img"
      aria-label="Your Expected Putts on This Green"
      viewBox={`0 0 ${W} ${H}`}
      className="w-full max-w-sm"
    >
      <polyline
        points={`${PAD},${PAD / 2} ${PAD},${H - PAD} ${W - PAD / 2},${H - PAD}`}
        fill="none"
        stroke="currentColor"
        strokeOpacity={0.4}
      />
      <polyline
        points={points.map((p) => p.join(",")).join(" ")}
        fill="none"
        stroke="var(--sidekick-color-accent)"
      />
      {points.map(([cx, cy], i) => (
        <circle
          key={i}
          cx={cx}
          cy={cy}
          r={2.5}
          fill="var(--sidekick-color-accent)"
        />
      ))}
      <circle
        cx={x(report.distance_ft)}
        cy={y(report.expected_putts)}
        r={4}
        fill="var(--sidekick-color-error)"
      />
      <text x={PAD + 4} y={PAD / 2 + 8} fontSize={9} fill="currentColor">
        Expected Putts (1–{maxE})
      </text>
      <text
        x={W - PAD / 2}
        y={H - 8}
        fontSize={9}
        fill="currentColor"
        textAnchor="end"
      >
        Distance (0–{shownFt(maxD, system).toFixed(0)} {unit})
      </text>
    </svg>
  );
}

export function StrokesGainedPanel() {
  const parameters = usePuttingStore((state) => state.parameters);
  const noise = usePuttingStore((state) => state.noise);
  const result = usePuttingStore((state) => state.result);
  const system = usePuttingStore((state) => state.unitSystem);
  const [entered, setEntered] = useState<typeof result>(null);
  const [holed, setHoled] = useState(false);
  const [leaveFt, setLeaveFt] = useState(0);
  const score = useMutation({ mutationFn: runStrokesGained });
  const unit = unitSuffix("distance", system).trim();
  const inputStyle = {
    backgroundColor: "var(--sidekick-color-input)",
    borderColor: "var(--sidekick-color-border)",
  };

  // Enter each new simulated putt, as the desktop's set_putt does.
  if (result && result !== entered) {
    setEntered(result);
    setHoled(result.holed);
    setLeaveFt(puttLeaveFt(result));
  }

  const leaveShown = shownFt(leaveFt, system);
  const feet = (shown: number) =>
    fromDisplay(shown, "distance", system) / FT_TO_M;
  return (
    <div className="space-y-3 text-xs">
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={holed}
            onChange={(event) => setHoled(event.target.checked)}
          />
          Holed
        </label>
        <label className="flex items-center gap-2">
          Leave ({unit})
          <input
            type="number"
            min={0}
            step={0.01}
            disabled={holed}
            value={leaveShown}
            onChange={(event) => setLeaveFt(feet(Number(event.target.value)))}
            className="w-24 rounded border px-2 py-1 disabled:opacity-60"
            style={inputStyle}
          />
        </label>
        <button
          type="button"
          disabled={score.isPending}
          onClick={() =>
            score.mutate(
              sgArgs(parameters, noise, { holed, leave_ft: feet(leaveShown) }),
            )
          }
          className="rounded border px-3 py-1 disabled:opacity-60"
          style={inputStyle}
        >
          Score Putt
        </button>
      </div>
      <p style={{ color: "var(--sidekick-color-text-muted)" }}>
        Scored against your own E(d), solved by the shared core on this planar
        green for the batch panel&apos;s player spread.
      </p>
      {score.isPending && <p>Solving your expected putts on this green…</p>}
      {score.error && (
        <p role="alert" style={{ color: "var(--sidekick-color-error)" }}>
          {score.error.message}
        </p>
      )}
      {score.data && (
        <>
          <p className="font-mono">{sgSummary(score.data, system)}</p>
          <CurveChart report={score.data} system={system} />
        </>
      )}
    </div>
  );
}
