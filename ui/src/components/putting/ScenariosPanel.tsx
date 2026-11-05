/** Saved scenarios and their comparison from the shared core (#14472). */
import { useMutation } from "@tanstack/react-query";
import { useState, type ButtonHTMLAttributes } from "react";

import { compareScenarios, runScenario } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import {
  MAX_SCENARIOS,
  SCENARIO_N,
  setupArgs,
  type Scenario,
} from "@/utils/puttingSetup";
import { formatQuantity } from "@/utils/units";

const FT_TO_M = 0.3048;
const muted = { color: "var(--sidekick-color-text-muted)" };
const field = {
  backgroundColor: "var(--sidekick-color-input)",
  borderColor: "var(--sidekick-color-border)",
};
const percent = (rate: number) => `${(rate * 100).toFixed(1)}%`;
const signed = (text: string) => (text.startsWith("-") ? text : `+${text}`);

function Action(props: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className="rounded border px-3 py-1 disabled:opacity-60"
      style={field}
      {...props}
    />
  );
}

export function ScenariosPanel() {
  const parameters = usePuttingStore((state) => state.parameters);
  const noise = usePuttingStore((state) => state.noise);
  const system = usePuttingStore((state) => state.unitSystem);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [name, setName] = useState("Scenario 1");
  const compare = useMutation({ mutationFn: compareScenarios });
  const save = useMutation({
    mutationFn: runScenario,
    onSuccess: (scenario) => {
      const next = [...scenarios, scenario].slice(-MAX_SCENARIOS);
      setScenarios(next);
      setName(`Scenario ${next.length + 1}`);
      compare.reset();
    },
  });
  const error = save.error ?? compare.error;
  const rows = compare.data?.rows;
  const leave = (ft: number | null) =>
    ft === null
      ? "All Holed"
      : formatQuantity(ft * FT_TO_M, "distance", system, 2);

  return (
    <div className="space-y-3 text-xs">
      <label className="block">
        <span style={muted}>Scenario Name</span>
        <input
          aria-label="Scenario Name"
          value={name}
          maxLength={80}
          onChange={(event) => setName(event.target.value)}
          className="mt-1 w-full rounded border px-2 py-1"
          style={field}
        />
      </label>
      <div className="flex flex-wrap gap-2">
        <Action
          disabled={save.isPending || !name.trim()}
          onClick={() =>
            save.mutate({
              name: name.trim(),
              setup: setupArgs(parameters, noise),
              n: SCENARIO_N,
              seed: parameters.random_seed,
            })
          }
        >
          Save Scenario
        </Action>
        <Action
          disabled={scenarios.length < 2 || compare.isPending}
          onClick={() => compare.mutate({ scenarios })}
        >
          Compare
        </Action>
      </div>
      <p style={muted}>
        Each scenario rolls {SCENARIO_N} putts of the shared core&apos;s skill
        model on this distance, slope, fall line, stimp and spread.
      </p>
      {save.isPending && <p>Rolling putts in the shared core…</p>}
      {error && (
        <p role="alert" style={{ color: "var(--sidekick-color-error)" }}>
          {error.message}
        </p>
      )}
      {scenarios.length > 0 && (
        <table
          aria-label={
            rows ? `Against ${compare.data?.baseline}` : "Saved Scenarios"
          }
          className="w-full font-mono"
        >
          <caption className="text-left" style={muted}>
            Make %, Mean Leave
            {rows && `, Change Against ${compare.data?.baseline}`}
          </caption>
          <tbody>
            {scenarios.map((scenario, i) => (
              <tr key={i}>
                <td className="max-w-0 truncate">{scenario.name}</td>
                <td className="text-right">
                  {percent(scenario.result.make_rate)}
                </td>
                <td className="text-right">
                  {leave(scenario.result.mean_leave_ft)}
                </td>
                {rows && (
                  <td className="text-right">
                    {signed(
                      `${(rows[i].make_rate_delta * 100).toFixed(1)} pts`,
                    )}
                  </td>
                )}
                {rows && (
                  <td className="text-right">
                    {rows[i].mean_leave_delta_ft === null
                      ? "—"
                      : signed(leave(rows[i].mean_leave_delta_ft))}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
