/**
 * Launch Monitor Analytics — Relationships panel (#11987, slice 4b).
 *
 * Web counterpart of the desktop "Relationships" tab
 * (`src/tools/launch_monitor_analytics/gui.py` `_build_relationships_tab` /
 * `_read_relationship_params` / `_compute_relationship` /
 * `_compute_multivariate`): pick a correlation method, metrics, optional
 * partial-correlation controls and a network edge threshold, then run the
 * same `compute_correlations` / `compute_pca` / `compute_vif` contracts
 * through `POST /v2/relationships` and `POST /v2/multivariate`. Shares the
 * CSV already loaded by `LaunchMonitorAnalyticsPage` — no separate upload
 * step.
 */

import { useCallback, useMemo, useState } from "react";
import {
  analyzeMultivariateV2,
  analyzeRelationshipsV2,
  type MultivariateResponse,
  type RelationshipMethod,
  type RelationshipsResponse,
} from "@/api/launchMonitorAnalytics";
import { numericColumns, type CsvValue } from "./LaunchMonitorAnalytics";

/** Same three choices as the desktop `relationship_method` combo. */
const METHOD_OPTIONS: RelationshipMethod[] = ["pearson", "spearman", "kendall"];

/** Matches `_read_relationship_params`'s `self.edge_threshold_spin` default. */
const DEFAULT_EDGE_THRESHOLD = 0.3;

const RUN_BUTTON_CLASS =
  "self-end rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium";

const DEFAULT_STATUS =
  "Select at least two metrics, then map interdependencies and run PCA/VIF.";

type RunState = "idle" | "running" | "done" | "error";
type RunLabel = "Relationship analysis" | "Multivariate analysis";

/**
 * Render a nullable statistic — never a misleading zero. Mirrors
 * `LaunchMonitorDispersionPanel`'s local `formatValue`: the API's
 * `_json_safe_float` never serializes an unavailable value as `0`, so the
 * UI must not either.
 */
function formatValue(value: number | null | undefined, digits = 4): string {
  if (value == null || Number.isNaN(value)) {
    return "unavailable";
  }
  return value.toFixed(digits);
}

/**
 * The selection to use: whichever of `choice` still names a current option,
 * derived during render rather than synced in effects (`set-state-in-effect`,
 * fixed for the Dispersion panel in #12076) so a column that leaves the CSV
 * drops out without a cascading re-render.
 */
function validSelection(choice: string[], options: string[]): string[] {
  return choice.filter((item) => options.includes(item));
}

function CoefficientMatrix({ result }: { result: RelationshipsResponse }) {
  return (
    <table className="w-full text-xs text-left">
      <thead>
        <tr className="text-gray-400 border-b border-gray-700">
          <th className="px-2 py-1">Metric</th>
          {result.metrics.map((metric) => (
            <th key={metric} className="px-2 py-1">
              {metric}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {result.metrics.map((metric, rowIndex) => (
          <tr key={metric} className="border-b border-gray-800 text-gray-200">
            <td className="px-2 py-1">{metric}</td>
            {result.coefficients[rowIndex].map((value, columnIndex) => (
              <td key={result.metrics[columnIndex]} className="px-2 py-1">
                {formatValue(value)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EdgesTable({ result }: { result: RelationshipsResponse }) {
  if (result.edges.length === 0) {
    return (
      <p className="text-xs text-gray-400">
        No pair cleared the edge threshold.
      </p>
    );
  }
  return (
    <table className="w-full text-xs text-left">
      <thead>
        <tr className="text-gray-400 border-b border-gray-700">
          <th className="px-2 py-1">Source</th>
          <th className="px-2 py-1">Target</th>
          <th className="px-2 py-1">r</th>
          <th className="px-2 py-1">p</th>
          <th className="px-2 py-1">adj. p</th>
          <th className="px-2 py-1">n</th>
          <th className="px-2 py-1">Flags</th>
        </tr>
      </thead>
      <tbody>
        {result.edges.map((edge) => (
          <tr
            key={`${edge.source}-${edge.target}`}
            className="border-b border-gray-800 text-gray-200"
          >
            <td className="px-2 py-1">{edge.source}</td>
            <td className="px-2 py-1">{edge.target}</td>
            <td className="px-2 py-1">{formatValue(edge.coefficient)}</td>
            <td className="px-2 py-1">{formatValue(edge.p_value)}</td>
            <td className="px-2 py-1">{formatValue(edge.adjusted_p_value)}</td>
            <td className="px-2 py-1">{edge.sample_count}</td>
            <td className="px-2 py-1">
              {edge.includes_derived_metric && "derived "}
              {edge.includes_boolean_projection && "0/1"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PCATable({ result }: { result: MultivariateResponse }) {
  const { pca } = result;
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-gray-400">n = {pca.sample_count}</p>
      <table className="w-full text-xs text-left">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="px-2 py-1">Metric</th>
            {pca.component_names.map((name) => (
              <th key={name} className="px-2 py-1">
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr className="border-b border-gray-800 text-gray-200">
            <td className="px-2 py-1">Explained Variance Ratio</td>
            {pca.explained_variance_ratio.map((value, index) => (
              <td key={pca.component_names[index]} className="px-2 py-1">
                {formatValue(value)}
              </td>
            ))}
          </tr>
          {pca.metrics.map((metric, rowIndex) => (
            <tr key={metric} className="border-b border-gray-800 text-gray-200">
              <td className="px-2 py-1">{metric}</td>
              {pca.loadings[rowIndex].map((value, columnIndex) => (
                <td key={pca.component_names[columnIndex]} className="px-2 py-1">
                  {formatValue(value)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VIFTable({ result }: { result: MultivariateResponse }) {
  const { vif } = result;
  const rows = Object.entries(vif.values);
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-gray-400">n = {vif.sample_count}</p>
      <table className="w-full text-xs text-left">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="px-2 py-1">Metric</th>
            <th className="px-2 py-1">VIF</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([metric, value]) => (
            <tr key={metric} className="border-b border-gray-800 text-gray-200">
              <td className="px-2 py-1">{metric}</td>
              <td className="px-2 py-1">{formatValue(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-400">
        VIF &gt;= 5: {vif.warning_metrics.join(", ") || "none"}
      </p>
    </div>
  );
}

export function LaunchMonitorRelationshipsPanel({
  columns,
  records,
}: {
  columns: string[];
  records: Record<string, CsvValue>[];
}) {
  const metricOptions = useMemo(
    () => numericColumns(columns, records),
    [columns, records],
  );

  const [method, setMethod] = useState<RelationshipMethod>("pearson");
  const [metricsChoice, setMetricsChoice] = useState<string[]>([]);
  const [controlsChoice, setControlsChoice] = useState<string[]>([]);
  const [edgeThreshold, setEdgeThreshold] = useState(DEFAULT_EDGE_THRESHOLD);

  const [runState, setRunState] = useState<RunState>("idle");
  const [runLabel, setRunLabel] = useState<RunLabel>("Relationship analysis");
  const [runError, setRunError] = useState<string | null>(null);
  const [relationships, setRelationships] =
    useState<RelationshipsResponse | null>(null);
  const [multivariate, setMultivariate] =
    useState<MultivariateResponse | null>(null);

  // Derived during render rather than synced in effects, so a column that
  // leaves the CSV falls back without a cascading re-render (same pattern as
  // `LaunchMonitorDispersionPanel`'s `effectiveCoordinate`).
  const selectedMetrics = validSelection(metricsChoice, metricOptions);
  const selectedControls = validSelection(controlsChoice, metricOptions);
  // Mirrors `_read_relationship_params`: a control that is also a selected
  // metric is dropped before the analysis runs, though it stays visible
  // (still selected) in the controls list itself.
  const effectiveControls = selectedControls.filter(
    (control) => !selectedMetrics.includes(control),
  );

  const canRun = selectedMetrics.length >= 2 && records.length >= 3;

  // Two actions, like the desktop's "Map Interdependencies" and "Run PCA and
  // VIF Diagnostics" buttons (`_run_relationship_async` /
  // `_run_multivariate_async`).
  const runAction = useCallback(
    <T,>(
      label: RunLabel,
      request: () => Promise<T>,
      present: (result: T) => void,
    ) => {
      if (!canRun) return;
      setRunState("running");
      setRunLabel(label);
      setRunError(null);
      void request()
        .then((result) => {
          present(result);
          setRunState("done");
        })
        .catch((err) => {
          setRunError(err instanceof Error ? err.message : `${label} failed`);
          setRunState("error");
        });
    },
    [canRun],
  );

  const handleRunRelationships = useCallback(() => {
    runAction(
      "Relationship analysis",
      () =>
        analyzeRelationshipsV2(
          records,
          selectedMetrics,
          method,
          effectiveControls,
          edgeThreshold,
        ),
      setRelationships,
    );
  }, [runAction, records, selectedMetrics, method, effectiveControls, edgeThreshold]);

  const handleRunMultivariate = useCallback(() => {
    runAction(
      "Multivariate analysis",
      () => analyzeMultivariateV2(records, selectedMetrics),
      setMultivariate,
    );
  }, [runAction, records, selectedMetrics]);

  const handleEdgeThresholdChange = useCallback((raw: string) => {
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return;
    setEdgeThreshold(Math.min(1, Math.max(0, parsed)));
  }, []);

  const statusMessage = useMemo(() => {
    if (runState === "running") return `Running ${runLabel.toLowerCase()}…`;
    if (runState === "error") {
      return runError
        ? `${runLabel} could not run: ${runError}`
        : `${runLabel} could not run.`;
    }
    if (runState === "done" && runLabel === "Multivariate analysis" && multivariate) {
      // Same wording as the desktop's `_present_multivariate` status line.
      const warning = multivariate.vif.warning_metrics.join(", ") || "none";
      return (
        `PCA/VIF complete for ${multivariate.pca.sample_count} complete shots. ` +
        `VIF >= 5: ${warning}.`
      );
    }
    if (runState === "done" && relationships) {
      return (
        `Mapped ${relationships.edges.length} screened dependency edge(s) ` +
        `across ${relationships.metrics.length} metrics.`
      );
    }
    return DEFAULT_STATUS;
  }, [runState, runLabel, runError, relationships, multivariate]);

  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 flex flex-col gap-3">
      <h2 className="text-sm font-medium text-white">Relationships</h2>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Method
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={method}
            onChange={(e) => setMethod(e.target.value as RelationshipMethod)}
          >
            {METHOD_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Metrics
          </span>
          <select
            multiple
            size={5}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={selectedMetrics}
            onChange={(e) =>
              setMetricsChoice(
                Array.from(e.target.selectedOptions).map((o) => o.value),
              )
            }
          >
            {metricOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Partial-Correlation Controls
          </span>
          <select
            multiple
            size={5}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={selectedControls}
            onChange={(e) =>
              setControlsChoice(
                Array.from(e.target.selectedOptions).map((o) => o.value),
              )
            }
          >
            {metricOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Network Edge Threshold
          </span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={edgeThreshold}
            onChange={(e) => handleEdgeThresholdChange(e.target.value)}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          />
        </label>

        <button
          type="button"
          onClick={handleRunRelationships}
          disabled={!canRun || runState === "running"}
          className={RUN_BUTTON_CLASS}
        >
          Map Interdependencies
        </button>
        <button
          type="button"
          onClick={handleRunMultivariate}
          disabled={!canRun || runState === "running"}
          className={RUN_BUTTON_CLASS}
        >
          Run PCA and VIF Diagnostics
        </button>
      </div>

      <p
        data-testid="lmr-status"
        className={
          runState === "error"
            ? "text-red-300"
            : runState === "done"
              ? "text-emerald-300"
              : "text-gray-300"
        }
      >
        {statusMessage}
      </p>

      {relationships && (
        <>
          <div data-testid="lmr-coefficients-table">
            <h3 className="text-xs font-medium text-white mb-1">
              {relationships.method} Correlation
            </h3>
            <div className="overflow-x-auto">
              <CoefficientMatrix result={relationships} />
            </div>
          </div>

          <div data-testid="lmr-edges-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Screened Dependency Edges
            </h3>
            <div className="overflow-x-auto">
              <EdgesTable result={relationships} />
            </div>
          </div>
        </>
      )}

      {multivariate && (
        <>
          <div data-testid="lmr-pca-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Principal-Component Analysis
            </h3>
            <div className="overflow-x-auto">
              <PCATable result={multivariate} />
            </div>
          </div>

          <div data-testid="lmr-vif-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Variance Inflation Factors
            </h3>
            <div className="overflow-x-auto">
              <VIFTable result={multivariate} />
            </div>
          </div>
        </>
      )}
    </section>
  );
}

export default LaunchMonitorRelationshipsPanel;
