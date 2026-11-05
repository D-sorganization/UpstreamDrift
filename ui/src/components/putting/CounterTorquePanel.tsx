/** Counter-torque stabilization explorer (#14470) over the core couple model. */
import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import type { CounterTorqueRequest } from "@/api/generated/types";
import { fetchCounterTorque, fetchPutterPresets } from "@/api/puttingClient";
import {
  COUNTER_TORQUE_CONTROLS,
  COUNTER_TORQUE_DEFAULTS,
  ctReadout,
  ctSummary,
  type CounterTorqueSamples,
} from "@/utils/counterTorque";

const W = 260;
const H = 140;
const PAD = 24;
// The studio's torque strip chart: series and colour.
const CURVES = [
  ["tau_right_in_nm", "#f97316", "Right"],
  ["tau_left_in_nm", "#38bdf8", "Left"],
  ["tau_net_nm", "#4ade80", "Net"],
  ["tau_off_required_nm", "#c084fc", "Off-Plane"],
] as const;
const inputStyle = {
  backgroundColor: "var(--sidekick-color-input)",
  borderColor: "var(--sidekick-color-border)",
};

function CoupleChart({ samples }: { samples: CounterTorqueSamples }) {
  const t = samples.t_s;
  const tEnd = t[t.length - 1] || 1;
  const x = (ti: number) => PAD + (ti / tEnd) * (W - 1.5 * PAD);
  // Symmetric axis spanning the largest couple, so no curve leaves the chart.
  const span = Math.max(...CURVES.flatMap(([k]) => samples[k].map(Math.abs)));
  const y = (v: number) => H / 2 - (v / (span || 1)) * (H / 2 - PAD / 2);
  return (
    <>
      <svg
        role="img"
        aria-label="Hand Couples Over the Stroke"
        viewBox={`0 0 ${W} ${H}`}
        className="w-full max-w-sm"
      >
        <polyline
          points={`${PAD},${PAD / 2} ${PAD},${H - PAD / 2} ${PAD},${H / 2} ${W - PAD / 2},${H / 2}`}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        {CURVES.map(([key, colour, name], i) => (
          <g key={key} stroke={colour} fill={colour} fontSize={8}>
            <polyline
              points={samples[key]
                .map((v, j) => `${x(t[j])},${y(v)}`)
                .join(" ")}
              fill="none"
            />
            <text x={PAD + 4 + i * 44} y={10} stroke="none">
              {name}
            </text>
          </g>
        ))}
      </svg>
      <p className="opacity-70">
        Hand couples (±{span.toFixed(1)} N·m) over the stroke
      </p>
    </>
  );
}

export function CounterTorquePanel() {
  const [request, setRequest] = useState<CounterTorqueRequest>(
    COUNTER_TORQUE_DEFAULTS,
  );
  const presets = useQuery({
    queryKey: ["putting", "putter-presets"],
    queryFn: fetchPutterPresets,
  });
  const run = useMutation({ mutationFn: fetchCounterTorque });
  const update = (changes: Partial<CounterTorqueRequest>) =>
    setRequest((current) => ({ ...current, ...changes }));

  return (
    <div className="space-y-3 text-xs">
      {COUNTER_TORQUE_CONTROLS.map((c) => (
        <label key={c.key} className="block">
          <span className="mb-1 flex justify-between">
            <span>{c.label}</span>
            <span className="font-mono">
              {request[c.key].toFixed(c.step < 1 ? 2 : 0)}
              {c.unit}
            </span>
          </span>
          <input
            aria-label={c.label}
            type="range"
            min={c.min}
            max={c.max}
            step={c.step}
            value={request[c.key]}
            onChange={(event) =>
              update({ [c.key]: Number(event.target.value) })
            }
            className="w-full"
            style={{ accentColor: "var(--sidekick-color-accent)" }}
          />
        </label>
      ))}
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex w-full min-w-0 items-center gap-2">
          Putter Head
          <select
            aria-label="Putter Head"
            value={request.putter}
            onChange={(event) => update({ putter: event.target.value })}
            className="min-w-0 flex-1 rounded border px-2 py-1"
            style={inputStyle}
          >
            {(presets.data?.presets ?? []).map((p) => (
              <option key={p.key} value={p.key}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          disabled={run.isPending}
          onClick={() => run.mutate(request)}
          className="rounded border px-3 py-1 disabled:opacity-60"
          style={inputStyle}
        >
          Run Counter-Torque
        </button>
      </div>
      {run.error && (
        <p role="alert" style={{ color: "var(--sidekick-color-error)" }}>
          {run.error.message}
        </p>
      )}
      {run.data && (
        <>
          {ctReadout(ctSummary(run.data.samples)).map((line) => (
            <p key={line} className="font-mono">
              {line}
            </p>
          ))}
          <CoupleChart samples={run.data.samples} />
        </>
      )}
    </div>
  );
}
