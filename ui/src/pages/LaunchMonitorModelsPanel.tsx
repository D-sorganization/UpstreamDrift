/**
 * Launch Monitor Analytics — Models panel (#11987, slice 5b).
 *
 * Web counterpart of the desktop "Models" tab
 * (`src/tools/launch_monitor_analytics/gui.py` `_build_models_tab` /
 * `_read_model_params` / `_compute_model`): pick a target, one or more
 * features, a model recipe, a random seed and an optional grouped holdout,
 * then run the same `fit_predictive_model` contract through
 * `POST /v2/model`. Shares the CSV already loaded by
 * `LaunchMonitorAnalyticsPage` — no separate upload step.
 */

import { useCallback, useMemo, useState } from "react";
import {
  fitModelV2,
  formatStat,
  type ModelGroupColumn,
  type ModelResponse,
  type PredictiveModelName,
} from "@/api/launchMonitorAnalytics";
import { numericColumns, type CsvValue } from "./LaunchMonitorAnalytics";

/** Same five choices as the desktop `model_type` combo. */
const MODEL_OPTIONS: PredictiveModelName[] = [
  "linear",
  "ridge",
  "lasso",
  "elastic_net",
  "mlp",
];
/** Matches `_read_model_params`'s `self.model_seed` default. */
const DEFAULT_SEED = 42;
const MAX_SEED = 2_147_483_647;
/** Sentinel for the desktop combo's "(random split)" item — no grouping. */
const RANDOM_SPLIT = "(random split)";
/**
 * Grouped-holdout candidates, matching the desktop combo's fixed item list
 * (`_build_models_tab`) and `ModelPayloadV2.group_column`'s literal union.
 * Only a candidate actually present in the loaded CSV is offered (same
 * reasoning as `LaunchMonitorDispersionPanel`'s `GROUP_CANDIDATES`).
 */
const GROUP_CANDIDATES: ModelGroupColumn[] = [
  "session_id",
  "monitor_vendor",
  "club",
];
/** Predictions beyond this count are not rendered (kept DOM-sized). */
const MAX_DISPLAYED_PREDICTIONS = 50;

const RUN_BUTTON_CLASS =
  "self-end rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium";

const DEFAULT_STATUS =
  "Select a target and one or more features, then fit and validate a model.";

type RunState = "idle" | "running" | "done" | "error" | "unavailable";

function MetricsTable({ metrics }: { metrics: Record<string, number | null> }) {
  const rows = Object.entries(metrics);
  return (
    <table className="w-full text-xs text-left">
      <thead>
        <tr className="text-gray-400 border-b border-gray-700">
          <th className="px-2 py-1">Metric</th>
          <th className="px-2 py-1">Value</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([name, value]) => (
          <tr key={name} className="border-b border-gray-800 text-gray-200">
            <td className="px-2 py-1">{name}</td>
            <td className="px-2 py-1">{formatStat(value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CoefficientsTable({ result }: { result: ModelResponse }) {
  if (!result.coefficients) {
    return (
      <p className="text-xs text-gray-400" data-testid="lmm-no-coefficients">
        Coefficients are not available for the {result.model} model.
      </p>
    );
  }
  const rows = Object.entries(result.coefficients);
  return (
    <table className="w-full text-xs text-left">
      <thead>
        <tr className="text-gray-400 border-b border-gray-700">
          <th className="px-2 py-1">Feature</th>
          <th className="px-2 py-1">Coefficient</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([feature, value]) => (
          <tr key={feature} className="border-b border-gray-800 text-gray-200">
            <td className="px-2 py-1">{feature}</td>
            <td className="px-2 py-1">{formatStat(value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PredictionsTable({ result }: { result: ModelResponse }) {
  const shown = result.predictions.slice(0, MAX_DISPLAYED_PREDICTIONS);
  return (
    <div className="flex flex-col gap-1">
      <table className="w-full text-xs text-left">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="px-2 py-1">Row</th>
            <th className="px-2 py-1">Actual</th>
            <th className="px-2 py-1">Predicted</th>
            <th className="px-2 py-1">Residual</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((row, index) => (
            <tr
              key={row.row_index ?? index}
              className="border-b border-gray-800 text-gray-200"
            >
              <td className="px-2 py-1">{formatStat(row.row_index, 0)}</td>
              <td className="px-2 py-1">{formatStat(row.actual)}</td>
              <td className="px-2 py-1">{formatStat(row.predicted)}</td>
              <td className="px-2 py-1">{formatStat(row.residual)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {result.predictions.length > MAX_DISPLAYED_PREDICTIONS && (
        <p className="text-xs text-gray-400">
          Showing the first {MAX_DISPLAYED_PREDICTIONS} of{" "}
          {result.predictions.length} predictions.
        </p>
      )}
    </div>
  );
}

export function LaunchMonitorModelsPanel({
  columns,
  records,
}: {
  columns: string[];
  records: Record<string, CsvValue>[];
}) {
  const numericOptions = useMemo(
    () => numericColumns(columns, records),
    [columns, records],
  );
  const groupOptions = useMemo(
    () => GROUP_CANDIDATES.filter((candidate) => columns.includes(candidate)),
    [columns],
  );

  const [target, setTarget] = useState("");
  const [featuresChoice, setFeaturesChoice] = useState<string[]>([]);
  const [model, setModel] = useState<PredictiveModelName>("linear");
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [groupChoice, setGroupChoice] = useState(RANDOM_SPLIT);

  const [runState, setRunState] = useState<RunState>("idle");
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<ModelResponse | null>(null);

  // Derived during render rather than synced in effects, so a column that
  // leaves the CSV falls back without a cascading re-render.
  const selectedTarget = numericOptions.includes(target) ? target : "";
  const featureOptions = numericOptions.filter(
    (column) => column !== selectedTarget,
  );
  const selectedFeatures = featuresChoice.filter((feature) =>
    featureOptions.includes(feature),
  );
  const groupColumn =
    groupChoice === RANDOM_SPLIT ||
    groupOptions.includes(groupChoice as ModelGroupColumn)
      ? groupChoice
      : RANDOM_SPLIT;

  const canRun =
    selectedTarget !== "" && selectedFeatures.length > 0 && records.length >= 3;

  const handleRun = useCallback(() => {
    if (!canRun) return;
    setRunState("running");
    setRunError(null);
    void fitModelV2(
      records,
      selectedTarget,
      selectedFeatures,
      model,
      seed,
      groupColumn === RANDOM_SPLIT ? null : (groupColumn as ModelGroupColumn),
    )
      .then((data) => {
        setResult(data);
        setRunState("done");
      })
      .catch((err) => {
        const message =
          err instanceof Error ? err.message : "Predictive model failed";
        setRunError(message);
        setRunState(
          message.toLowerCase().includes("unavailable")
            ? "unavailable"
            : "error",
        );
      });
  }, [canRun, records, selectedTarget, selectedFeatures, model, seed, groupColumn]);

  const handleSeedChange = useCallback((raw: string) => {
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return;
    setSeed(Math.min(MAX_SEED, Math.max(0, Math.trunc(parsed))));
  }, []);

  const statusMessage = useMemo(() => {
    if (runState === "running") return "Fitting and validating the model…";
    if (runState === "unavailable") {
      const reason =
        runError ?? "the selected model's optional dependency is missing.";
      return `Model unavailable: ${reason}`;
    }
    if (runState === "error") {
      return runError
        ? `Predictive model could not run: ${runError}`
        : "Predictive model could not run.";
    }
    if (runState === "done" && result) {
      return (
        `Model complete: R2=${formatStat(result.metrics.r2, 3)}, ` +
        `RMSE=${formatStat(result.metrics.rmse, 3)}.`
      );
    }
    return DEFAULT_STATUS;
  }, [runState, runError, result]);

  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 flex flex-col gap-3">
      <h2 className="text-sm font-medium text-white">Models</h2>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Target
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={selectedTarget}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="">(select target)</option>
            {numericOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Features
          </span>
          <select
            multiple
            size={5}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={selectedFeatures}
            onChange={(e) =>
              setFeaturesChoice(
                Array.from(e.target.selectedOptions).map((o) => o.value),
              )
            }
          >
            {featureOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Model
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={model}
            onChange={(e) => setModel(e.target.value as PredictiveModelName)}
          >
            {MODEL_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Grouped Holdout
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={groupColumn}
            onChange={(e) => setGroupChoice(e.target.value)}
          >
            <option value={RANDOM_SPLIT}>{RANDOM_SPLIT}</option>
            {groupOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Random Seed
          </span>
          <input
            type="number"
            min={0}
            max={MAX_SEED}
            value={seed}
            onChange={(e) => handleSeedChange(e.target.value)}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          />
        </label>

        <button
          type="button"
          onClick={handleRun}
          disabled={!canRun || runState === "running"}
          className={RUN_BUTTON_CLASS}
        >
          Fit and Validate Model
        </button>
      </div>

      <p
        data-testid="lmm-status"
        className={
          runState === "error"
            ? "text-red-300"
            : runState === "unavailable"
              ? "text-amber-300"
              : runState === "done"
                ? "text-emerald-300"
                : "text-gray-300"
        }
      >
        {statusMessage}
      </p>

      {result && (
        <>
          <div data-testid="lmm-metrics-table">
            <h3 className="text-xs font-medium text-white mb-1">Metrics</h3>
            <div className="overflow-x-auto">
              <MetricsTable metrics={result.metrics} />
            </div>
          </div>

          <div data-testid="lmm-coefficients-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Coefficients
            </h3>
            <div className="overflow-x-auto">
              <CoefficientsTable result={result} />
            </div>
          </div>

          <p className="text-xs text-gray-400" data-testid="lmm-split-counts">
            Train n = {result.train_count} · Test n = {result.test_count}
          </p>

          <div data-testid="lmm-predictions-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Held-Out Predictions
            </h3>
            <div className="overflow-x-auto">
              <PredictionsTable result={result} />
            </div>
          </div>
        </>
      )}
    </section>
  );
}

export default LaunchMonitorModelsPanel;
