/** Saved scenarios, comparison and JSON setup files from the shared core (#14472). */
import { useMutation } from "@tanstack/react-query";
import { useState, type ButtonHTMLAttributes } from "react";

import {
  compareScenarios,
  exportSetup,
  importSetup,
  runScenario,
} from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import {
  MAX_SCENARIOS,
  SCENARIO_N,
  applySetup,
  setupArgs,
  setupFileName,
  type Scenario,
} from "@/utils/puttingSetup";
import { saveBlob, type BlobDownload } from "@/utils/saveBlob";
import { FT_TO_M, formatQuantity } from "@/utils/units";

const muted = { color: "var(--sidekick-color-text-muted)" };
const field = {
  backgroundColor: "var(--sidekick-color-input)",
  borderColor: "var(--sidekick-color-border)",
};
const percent = (rate: number) => `${(rate * 100).toFixed(1)}%`;
const signed = (text: string) => (text.startsWith("-") ? text : `+${text}`);

const BUTTON = "rounded border px-3 py-1 disabled:opacity-60";

function Action(props: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type="button" className={BUTTON} style={field} {...props} />;
}

async function readJson(file: File): Promise<unknown> {
  try {
    return JSON.parse(await file.text());
  } catch {
    throw new Error(`${file.name} is not a JSON file`);
  }
}

export function ScenariosPanel({
  download = saveBlob,
}: {
  download?: BlobDownload;
}) {
  const parameters = usePuttingStore((state) => state.parameters);
  const noise = usePuttingStore((state) => state.noise);
  const system = usePuttingStore((state) => state.unitSystem);
  const updateParameters = usePuttingStore((state) => state.updateParameters);
  const setNoise = usePuttingStore((state) => state.setNoise);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [name, setName] = useState("Scenario 1");
  const [studioRows, setStudioRows] = useState(0);
  const compare = useMutation({ mutationFn: compareScenarios });
  const keep = (next: Scenario[]) => {
    setScenarios(next);
    setName(`Scenario ${next.length + 1}`);
    compare.reset();
  };
  const save = useMutation({
    mutationFn: runScenario,
    onSuccess: (scenario) =>
      keep([...scenarios, scenario].slice(-MAX_SCENARIOS)),
  });
  const exporter = useMutation({
    mutationFn: exportSetup,
    onSuccess: (doc) =>
      download(
        new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" }),
        setupFileName(),
      ),
  });
  const importer = useMutation({
    mutationFn: async (file: File) =>
      importSetup({ document: await readJson(file) }),
    onSuccess: (imported) => {
      const loaded = applySetup(parameters, noise, imported.setup);
      updateParameters(loaded.parameters);
      setNoise(loaded.noise);
      setStudioRows(imported.studio_scenarios);
      keep(imported.scenarios);
    },
  });
  const error = save.error ?? compare.error ?? exporter.error ?? importer.error;
  const busy = save.isPending || importer.isPending;
  const rows = compare.data?.rows;
  const leave = (ft: number | null) =>
    ft === null
      ? "All Holed"
      : formatQuantity(ft * FT_TO_M, "distance", system, 2);

  return (
    <div className="space-y-3 pb-12 text-xs">
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
          disabled={busy || !name.trim()}
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
        <Action
          disabled={exporter.isPending}
          onClick={() =>
            exporter.mutate({ setup: setupArgs(parameters, noise), scenarios })
          }
        >
          Export JSON
        </Action>
        <label className={`${BUTTON} cursor-pointer`} style={field}>
          Import JSON
          <input
            aria-label="Import JSON"
            type="file"
            accept=".json,application/json"
            className="sr-only"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) importer.mutate(file);
              event.target.value = "";
            }}
          />
        </label>
      </div>
      <p style={muted}>
        Each scenario rolls {SCENARIO_N} putts of the shared core&apos;s skill
        model on this distance, slope, fall line, stimp and spread. Files use
        the core&apos;s setup schema, so the desktop and studio load them too.
      </p>
      {busy && <p>Rolling putts in the shared core…</p>}
      {error && (
        <p role="alert" style={{ color: "var(--sidekick-color-error)" }}>
          {error.message}
        </p>
      )}
      {studioRows > 0 && (
        <p style={muted}>
          {studioRows} studio scenario rows are text only, so only the
          studio&apos;s green and distance were loaded.
        </p>
      )}
      {scenarios.length > 0 && (
        <table
          aria-label={
            rows ? `Against ${compare.data?.baseline}` : "Saved Scenarios"
          }
          className="w-full font-mono"
        >
          <caption className="pb-1 text-left" style={muted}>
            Make % and Mean Leave
            {rows && `, Change Against ${compare.data?.baseline} Below`}
          </caption>
          <tbody>
            {scenarios.map((scenario, i) => (
              <tr key={i} className="align-top">
                <td className="w-full max-w-0 truncate">{scenario.name}</td>
                <td className="whitespace-nowrap pl-2 text-right">
                  {percent(scenario.result.make_rate)}
                  {rows && (
                    <div style={muted}>
                      {signed(
                        `${(rows[i].make_rate_delta * 100).toFixed(1)} pts`,
                      )}
                    </div>
                  )}
                </td>
                <td className="whitespace-nowrap pl-2 text-right">
                  {leave(scenario.result.mean_leave_ft)}
                  {rows && (
                    <div style={muted}>
                      {rows[i].mean_leave_delta_ft === null
                        ? "—"
                        : signed(leave(rows[i].mean_leave_delta_ft))}
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
