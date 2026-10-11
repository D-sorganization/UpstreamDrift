/**
 * Launch Monitor Analytics — Monitor Comparison panel (#11987, slice 5b).
 *
 * Web counterpart of the desktop "Monitor Comparison" tab
 * (`src/tools/launch_monitor_analytics/gui.py` `_build_comparison_tab` /
 * `_read_comparison_params` / `_compute_comparison`): pick a metric, an
 * optional matched-shot column, and an optional reference monitor, then run
 * the same `compare_monitors` contract through `POST /v2/comparison`. Shares
 * the CSV already loaded by `LaunchMonitorAnalyticsPage` — no separate
 * upload step.
 */

import { useCallback, useMemo, useState } from "react";
import {
  compareMonitorsV2,
  formatStat,
  type ComparisonResponse,
} from "@/api/launchMonitorAnalytics";
import { numericColumns, type CsvValue } from "./LaunchMonitorAnalytics";

/** Sentinel for the desktop combo's "(unmatched)" item — no matched-shot key. */
const UNMATCHED = "(unmatched)";
/** Sentinel for the reference combo's blank state (`currentText() or None`). */
const DEFAULT_REFERENCE = "(default: first monitor)";

const RUN_BUTTON_CLASS =
  "self-end rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium";

const DEFAULT_STATUS =
  "Select a metric, then compare monitor behavior across the loaded shots.";

type RunState = "idle" | "running" | "done" | "error";

/** Distinct `monitor_vendor` values present in `records`, sorted (desktop parity: `_refresh_all`). */
function monitorVendors(records: Record<string, CsvValue>[]): string[] {
  const seen = new Set<string>();
  for (const record of records) {
    const value = record.monitor_vendor;
    if (typeof value === "string" && value !== "") seen.add(value);
  }
  return Array.from(seen).sort();
}

function SummariesTable({ result }: { result: ComparisonResponse }) {
  return (
    <table className="w-full text-xs text-left">
      <thead>
        <tr className="text-gray-400 border-b border-gray-700">
          <th className="px-2 py-1">Monitor</th>
          <th className="px-2 py-1">n</th>
          <th className="px-2 py-1">Mean</th>
          <th className="px-2 py-1">Std Dev</th>
          <th className="px-2 py-1">Median</th>
        </tr>
      </thead>
      <tbody>
        {result.summaries.map((summary) => (
          <tr
            key={summary.monitor}
            className="border-b border-gray-800 text-gray-200"
          >
            <td className="px-2 py-1">{summary.monitor}</td>
            <td className="px-2 py-1">{summary.sample_count}</td>
            <td className="px-2 py-1">{formatStat(summary.mean)}</td>
            <td className="px-2 py-1">
              {formatStat(summary.standard_deviation)}
            </td>
            <td className="px-2 py-1">{formatStat(summary.median)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PairwiseTable({ result }: { result: ComparisonResponse }) {
  return (
    <table className="w-full text-xs text-left">
      <thead>
        <tr className="text-gray-400 border-b border-gray-700">
          <th className="px-2 py-1">Reference</th>
          <th className="px-2 py-1">Comparator</th>
          <th className="px-2 py-1">Matched</th>
          <th className="px-2 py-1">n</th>
          <th className="px-2 py-1">Mean Bias</th>
          <th className="px-2 py-1">Std Dev Bias</th>
          <th className="px-2 py-1">Lower Limit</th>
          <th className="px-2 py-1">Upper Limit</th>
          <th className="px-2 py-1">Slope</th>
          <th className="px-2 py-1">Intercept</th>
          <th className="px-2 py-1">Correlation</th>
          <th className="px-2 py-1">Warning</th>
        </tr>
      </thead>
      <tbody>
        {result.pairwise.map((item) => (
          <tr
            key={`${item.reference}-${item.comparator}`}
            className="border-b border-gray-800 text-gray-200"
          >
            <td className="px-2 py-1">{item.reference}</td>
            <td className="px-2 py-1">{item.comparator}</td>
            <td className="px-2 py-1">{item.matched ? "yes" : "no"}</td>
            <td className="px-2 py-1">{item.sample_count}</td>
            <td className="px-2 py-1">{formatStat(item.mean_bias)}</td>
            <td className="px-2 py-1">
              {formatStat(item.standard_deviation_bias)}
            </td>
            <td className="px-2 py-1">{formatStat(item.lower_limit)}</td>
            <td className="px-2 py-1">{formatStat(item.upper_limit)}</td>
            <td className="px-2 py-1">{formatStat(item.slope)}</td>
            <td className="px-2 py-1">{formatStat(item.intercept)}</td>
            <td className="px-2 py-1">{formatStat(item.correlation)}</td>
            <td className="px-2 py-1 text-amber-300">{item.warning ?? ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function LaunchMonitorComparisonPanel({
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
  const referenceOptions = useMemo(() => monitorVendors(records), [records]);

  const [metric, setMetric] = useState("");
  const [matchChoice, setMatchChoice] = useState(UNMATCHED);
  const [referenceChoice, setReferenceChoice] = useState(DEFAULT_REFERENCE);

  const [runState, setRunState] = useState<RunState>("idle");
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<ComparisonResponse | null>(null);

  // Derived during render rather than synced in effects, so a column/vendor
  // that leaves the CSV falls back without a cascading re-render (same
  // pattern as the Dispersion/Relationships panels).
  const selectedMetric = metricOptions.includes(metric) ? metric : "";
  const matchColumn =
    matchChoice !== UNMATCHED && columns.includes(matchChoice)
      ? matchChoice
      : UNMATCHED;
  const referenceMonitor =
    referenceChoice !== DEFAULT_REFERENCE &&
    referenceOptions.includes(referenceChoice)
      ? referenceChoice
      : DEFAULT_REFERENCE;

  const canRun = selectedMetric !== "" && records.length >= 3;

  const handleRun = useCallback(() => {
    if (!canRun) return;
    setRunState("running");
    setRunError(null);
    void compareMonitorsV2(
      records,
      selectedMetric,
      matchColumn === UNMATCHED ? null : matchColumn,
      referenceMonitor === DEFAULT_REFERENCE ? null : referenceMonitor,
    )
      .then((data) => {
        setResult(data);
        setRunState("done");
      })
      .catch((err) => {
        setRunError(
          err instanceof Error ? err.message : "Monitor comparison failed",
        );
        setRunState("error");
      });
  }, [canRun, records, selectedMetric, matchColumn, referenceMonitor]);

  const statusMessage = useMemo(() => {
    if (runState === "running") return "Comparing monitor behavior…";
    if (runState === "error") {
      return runError
        ? `Monitor comparison could not run: ${runError}`
        : "Monitor comparison could not run.";
    }
    if (runState === "done" && result) {
      const warning = result.pairwise.find((item) => item.warning)?.warning;
      return warning ?? "Matched monitor agreement analysis complete.";
    }
    return DEFAULT_STATUS;
  }, [runState, runError, result]);

  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 flex flex-col gap-3">
      <h2 className="text-sm font-medium text-white">Monitor Comparison</h2>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Metric
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={selectedMetric}
            onChange={(e) => setMetric(e.target.value)}
          >
            <option value="">(select metric)</option>
            {metricOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Matched-Shot Column
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={matchColumn}
            onChange={(e) => setMatchChoice(e.target.value)}
          >
            <option value={UNMATCHED}>{UNMATCHED}</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Reference Monitor
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={referenceMonitor}
            onChange={(e) => setReferenceChoice(e.target.value)}
          >
            <option value={DEFAULT_REFERENCE}>{DEFAULT_REFERENCE}</option>
            {referenceOptions.map((monitor) => (
              <option key={monitor} value={monitor}>
                {monitor}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={handleRun}
          disabled={!canRun || runState === "running"}
          className={RUN_BUTTON_CLASS}
        >
          Compare Monitor Behavior
        </button>
      </div>

      <p className="text-xs text-gray-400">
        Use a match identifier for the same shots whenever possible. Unmatched
        results are descriptive, not calibration evidence.
      </p>

      <p
        data-testid="lmc-status"
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

      {result && (
        <>
          <div data-testid="lmc-summaries-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Monitor Summaries
            </h3>
            <div className="overflow-x-auto">
              <SummariesTable result={result} />
            </div>
          </div>

          <div data-testid="lmc-pairwise-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Pairwise Agreement
            </h3>
            <div className="overflow-x-auto">
              <PairwiseTable result={result} />
            </div>
          </div>
        </>
      )}
    </section>
  );
}

export default LaunchMonitorComparisonPanel;
