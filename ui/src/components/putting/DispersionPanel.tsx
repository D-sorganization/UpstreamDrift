/** Batch dispersion and make-% spread from the shared putting core (#14460). */
import { useMutation } from "@tanstack/react-query";

import { runDispersion, runMakeSpread } from "@/api/puttingClient";
import type { DispersionReport, MakeSpreadReport } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import {
  BATCH_SIZES,
  NOISE_CHANNELS,
  SPREAD_N,
  dispersionArgs,
  type NoiseChannel,
} from "@/utils/dispersion";
import {
  formatQuantity,
  toDisplay,
  unitSuffix,
  type UnitSystem,
} from "@/utils/units";

const FT_TO_M = 0.3048;
const MM_PER_IN = 25.4;
const PLOT = 220; // SVG viewBox side
const PAD = 28;
const SPREAD_MAX_FT = 45;

/** Strike is shown in inches in imperial, like the desktop panel. */
function shown(channel: NoiseChannel, system: UnitSystem) {
  return channel.key === "strike_toe_sd_mm" && system === "imperial"
    ? { perShown: MM_PER_IN, suffix: "in" }
    : { perShown: channel.perShown, suffix: channel.suffix };
}

const ft = (valueFt: number, system: UnitSystem) =>
  toDisplay(valueFt * FT_TO_M, "distance", system);

function RestScatter({
  report,
  system,
}: {
  report: DispersionReport;
  system: UnitSystem;
}) {
  const reach = Math.max(
    1,
    ...report.rest_x_ft.map(Math.abs),
    ...report.rest_y_ft.map(Math.abs),
  );
  const scale = (PLOT / 2 - PAD / 2) / reach;
  const unit = unitSuffix("distance", system).trim();
  return (
    <svg
      role="img"
      aria-label={`Rest Positions Around the Hole (${report.n} Putts)`}
      viewBox={`0 0 ${PLOT} ${PLOT}`}
      className="w-full max-w-xs"
    >
      <line
        x1={0}
        y1={PLOT / 2}
        x2={PLOT}
        y2={PLOT / 2}
        stroke="currentColor"
        strokeOpacity={0.25}
      />
      <line
        x1={PLOT / 2}
        y1={0}
        x2={PLOT / 2}
        y2={PLOT}
        stroke="currentColor"
        strokeOpacity={0.25}
      />
      <circle
        cx={PLOT / 2}
        cy={PLOT / 2}
        r={4}
        fill="none"
        stroke="currentColor"
      />
      {report.rest_x_ft.map((x, i) => (
        <circle
          key={i}
          data-holed={report.holed[i]}
          cx={PLOT / 2 - report.rest_y_ft[i] * scale}
          cy={PLOT / 2 - x * scale}
          r={2}
          fill={
            report.holed[i]
              ? "var(--sidekick-color-success)"
              : "var(--sidekick-color-accent)"
          }
          fillOpacity={0.7}
        />
      ))}
      <text x={4} y={12} fontSize={9} fill="currentColor">
        Past Hole ↑ · Edge {ft(reach, system).toFixed(1)} {unit}
      </text>
    </svg>
  );
}

function SpreadChart({
  report,
  system,
}: {
  report: MakeSpreadReport;
  system: UnitSystem;
}) {
  const x = (d: number) => PAD + (d / SPREAD_MAX_FT) * (PLOT - 1.5 * PAD);
  const y = (rate: number) => PLOT - PAD - rate * (PLOT - 1.5 * PAD);
  const points = report.distances_ft.map((d, i) => [
    x(d),
    y(report.make_rate[i]),
  ]);
  const unit = unitSuffix("distance", system).trim();
  return (
    <svg
      role="img"
      aria-label="Make Rate by Distance (0–45 ft)"
      viewBox={`0 0 ${PLOT} ${PLOT}`}
      className="w-full max-w-xs"
    >
      <polyline
        points={`${PAD},${PAD / 2} ${PAD},${PLOT - PAD} ${PLOT - PAD / 2},${PLOT - PAD}`}
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
      <text x={PAD + 4} y={PAD / 2 + 8} fontSize={9} fill="currentColor">
        Make % (0–100)
      </text>
      <text
        x={PLOT - PAD / 2}
        y={PLOT - 8}
        fontSize={9}
        fill="currentColor"
        textAnchor="end"
      >
        Distance (0–{ft(SPREAD_MAX_FT, system).toFixed(0)} {unit})
      </text>
    </svg>
  );
}

export function DispersionPanel() {
  const parameters = usePuttingStore((state) => state.parameters);
  const system = usePuttingStore((state) => state.unitSystem);
  const noise = usePuttingStore((state) => state.noise);
  const setNoise = usePuttingStore((state) => state.setNoise);
  const batch = useMutation({ mutationFn: runDispersion });
  const spread = useMutation({ mutationFn: runMakeSpread });
  const busy = batch.isPending || spread.isPending;
  const error = batch.error ?? spread.error;
  const buttonStyle = {
    backgroundColor: "var(--sidekick-color-input)",
    borderColor: "var(--sidekick-color-border)",
  };

  return (
    <div className="space-y-3 text-xs">
      <div className="grid grid-cols-2 gap-2">
        {NOISE_CHANNELS.map((channel) => {
          const { perShown, suffix } = shown(channel, system);
          const label = `${channel.label} (${suffix})`;
          return (
            <label key={channel.key} className="block">
              <span style={{ color: "var(--sidekick-color-text-muted)" }}>
                {label}
              </span>
              <input
                aria-label={label}
                type="number"
                min={channel.min / perShown}
                max={channel.max / perShown}
                step="any"
                value={Number((noise[channel.key] / perShown).toFixed(4))}
                onChange={(event) =>
                  setNoise({
                    ...noise,
                    [channel.key]: Number(event.target.value) * perShown,
                  })
                }
                className="mt-1 w-full rounded border px-2 py-1"
                style={buttonStyle}
              />
            </label>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-2">
        {BATCH_SIZES.map((n) => (
          <button
            key={n}
            type="button"
            disabled={busy}
            onClick={() => batch.mutate(dispersionArgs(parameters, noise, n))}
            className="rounded border px-3 py-1 disabled:opacity-60"
            style={buttonStyle}
          >
            {n} Putts
          </button>
        ))}
        <button
          type="button"
          disabled={busy}
          onClick={() =>
            spread.mutate(dispersionArgs(parameters, noise, SPREAD_N))
          }
          className="rounded border px-3 py-1 disabled:opacity-60"
          style={buttonStyle}
        >
          0–45 ft Make-% Spread
        </button>
      </div>
      <p style={{ color: "var(--sidekick-color-text-muted)" }}>
        Batches roll the shared core&apos;s skill model on a planar green with
        this distance, slope, fall line and stimp. Grain, bumps, friction
        variation, the putter head and the impact offset are not part of the
        batch, so the scatter is the player&apos;s spread, not this 3D
        putt&apos;s.
      </p>
      {busy && <p>Rolling putts in the shared core…</p>}
      {error && (
        <p role="alert" style={{ color: "var(--sidekick-color-error)" }}>
          {error.message}
        </p>
      )}
      {batch.data && (
        <>
          <dl className="grid grid-cols-2 gap-x-3 font-mono">
            <dt style={{ color: "var(--sidekick-color-text-muted)" }}>
              Make Rate
            </dt>
            <dd className="text-right">
              {(batch.data.make_rate * 100).toFixed(1)}%
            </dd>
            <dt style={{ color: "var(--sidekick-color-text-muted)" }}>
              Mean Miss Leave
            </dt>
            <dd className="text-right">
              {batch.data.mean_leave_ft === null
                ? "All Holed"
                : formatQuantity(
                    batch.data.mean_leave_ft * FT_TO_M,
                    "distance",
                    system,
                    2,
                  )}
            </dd>
          </dl>
          <RestScatter report={batch.data} system={system} />
        </>
      )}
      {spread.data && <SpreadChart report={spread.data} system={system} />}
    </div>
  );
}
