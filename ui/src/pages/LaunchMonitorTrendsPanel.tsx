/**
 * Launch Monitor Analytics — Trends panel (#11987, slice 2b).
 *
 * Web counterpart of the desktop "Trends" tab
 * (`src/tools/launch_monitor_analytics/gui.py` `_build_trends_tab` /
 * `_read_trend_params` / `_compute_trend`): pick a metric, a time column, and
 * a rolling window, then run the same `analyze_trend` contract through
 * `POST /v2/trend`. Shares the CSV already loaded by
 * `LaunchMonitorAnalyticsPage` — no separate upload step.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  formatStat,
  postTrend,
  type TrendResponse,
} from "@/api/launchMonitorAnalytics";
import { numericColumns, type CsvValue } from "./LaunchMonitorAnalytics";

/** Minimum rolling window, matching the PyQt spinbox and `TrendPayloadV2`. */
const MIN_ROLLING_WINDOW = 3;
/** Maximum rolling window, matching the PyQt spinbox and `TrendPayloadV2`. */
const MAX_ROLLING_WINDOW = 500;
/** Default rolling window, matching the PyQt spinbox and `TrendPayloadV2`. */
const DEFAULT_ROLLING_WINDOW = 10;

const DEFAULT_STATUS =
  "Select a metric and time column, then run the longitudinal trend.";

type RunState = "idle" | "running" | "done" | "error";

/**
 * Time-column candidates, mirroring the desktop tab's `_refresh_all` filter
 * (`"time" in column or "date" in column or column == "captured_at"`).
 */
function timeColumnCandidates(columns: string[]): string[] {
  return columns.filter(
    (column) =>
      column.includes("time") ||
      column.includes("date") ||
      column === "captured_at",
  );
}

function ChangeCandidatesTable({ result }: { result: TrendResponse }) {
  if (result.change_candidates.length === 0) {
    return (
      <p className="text-xs text-gray-400">
        No candidate change points were found.
      </p>
    );
  }
  return (
    <table className="w-full text-xs text-left">
      <thead>
        <tr className="text-gray-400 border-b border-gray-700">
          <th className="px-2 py-1">Captured At</th>
          <th className="px-2 py-1">Row</th>
          <th className="px-2 py-1">Before mean</th>
          <th className="px-2 py-1">After mean</th>
          <th className="px-2 py-1">Effect size</th>
        </tr>
      </thead>
      <tbody>
        {result.change_candidates.map((candidate) => (
          <tr
            key={`${candidate.row_index}-${candidate.captured_at}`}
            className="border-b border-gray-800 text-gray-200"
          >
            <td className="px-2 py-1">{candidate.captured_at}</td>
            <td className="px-2 py-1">{candidate.row_index}</td>
            <td className="px-2 py-1">{formatStat(candidate.before_mean)}</td>
            <td className="px-2 py-1">{formatStat(candidate.after_mean)}</td>
            <td className="px-2 py-1">{formatStat(candidate.effect_size)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RollingTable({
  result,
  timeColumn,
}: {
  result: TrendResponse;
  timeColumn: string;
}) {
  if (result.rolling.length === 0) {
    return <p className="text-xs text-gray-400">No rolling series.</p>;
  }
  return (
    <div className="max-h-64 overflow-y-auto">
      <table className="w-full text-xs text-left">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="px-2 py-1">{timeColumn}</th>
            <th className="px-2 py-1">Value</th>
            <th className="px-2 py-1">Rolling mean</th>
            <th className="px-2 py-1">Rolling median</th>
            <th className="px-2 py-1">Rolling std</th>
            <th className="px-2 py-1">EWMA</th>
          </tr>
        </thead>
        <tbody>
          {result.rolling.map((point, index) => (
            <tr
              key={`${index}-${String(point[timeColumn])}`}
              className="border-b border-gray-800 text-gray-200"
            >
              <td className="px-2 py-1">{String(point[timeColumn] ?? "—")}</td>
              <td className="px-2 py-1">{formatStat(point.value)}</td>
              <td className="px-2 py-1">{formatStat(point.rolling_mean)}</td>
              <td className="px-2 py-1">{formatStat(point.rolling_median)}</td>
              <td className="px-2 py-1">{formatStat(point.rolling_std)}</td>
              <td className="px-2 py-1">{formatStat(point.ewma)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function LaunchMonitorTrendsPanel({
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
  const timeColumnOptions = useMemo(
    () => timeColumnCandidates(columns),
    [columns],
  );

  const [metric, setMetric] = useState("");
  const [timeColumn, setTimeColumn] = useState("");
  const [rollingWindowText, setRollingWindowText] = useState(
    String(DEFAULT_ROLLING_WINDOW),
  );
  const rollingWindow =
    rollingWindowText.trim() === "" ? Number.NaN : Number(rollingWindowText);

  const [runState, setRunState] = useState<RunState>("idle");
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<TrendResponse | null>(null);

  // Drop selections a newly loaded CSV no longer supports.
  useEffect(() => {
    setMetric((prev) => (metricOptions.includes(prev) ? prev : ""));
  }, [metricOptions]);
  useEffect(() => {
    setTimeColumn((prev) =>
      timeColumnOptions.includes(prev) ? prev : (timeColumnOptions[0] ?? ""),
    );
  }, [timeColumnOptions]);

  // Same bounds `TrendPayloadV2` enforces; out-of-range input disables Run
  // rather than being silently rewritten while the user types.
  const rollingWindowValid =
    Number.isInteger(rollingWindow) &&
    rollingWindow >= MIN_ROLLING_WINDOW &&
    rollingWindow <= MAX_ROLLING_WINDOW;
  const canRun =
    metric !== "" &&
    timeColumn !== "" &&
    rollingWindowValid &&
    records.length >= 3;

  const handleRun = useCallback(() => {
    if (!canRun) return;
    setRunState("running");
    setRunError(null);
    void postTrend(records, metric, timeColumn, rollingWindow)
      .then((data) => {
        setResult(data);
        setRunState("done");
      })
      .catch((err) => {
        setRunError(
          err instanceof Error ? err.message : "Trend analysis failed",
        );
        setRunState("error");
      });
  }, [canRun, records, metric, timeColumn, rollingWindow]);

  const statusMessage = useMemo(() => {
    if (runState === "running") return "Running trend analysis…";
    if (runState === "error") {
      return runError
        ? `Trend analysis could not run: ${runError}`
        : "Trend analysis could not run.";
    }
    if (runState === "done" && result) {
      return (
        `Trend slope=${formatStat(result.slope_per_day)}/day; ` +
        `${result.change_candidates.length} candidate change point(s).`
      );
    }
    return DEFAULT_STATUS;
  }, [runState, runError, result]);

  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 flex flex-col gap-3">
      <h2 className="text-sm font-medium text-white">Trends</h2>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Metric
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={metric}
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
            Time Column
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={timeColumn}
            onChange={(e) => setTimeColumn(e.target.value)}
          >
            <option value="">(select time column)</option>
            {timeColumnOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Rolling Window
          </span>
          <input
            type="number"
            min={MIN_ROLLING_WINDOW}
            max={MAX_ROLLING_WINDOW}
            value={rollingWindowText}
            onChange={(e) => setRollingWindowText(e.target.value)}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          />
        </label>

        <button
          type="button"
          onClick={handleRun}
          disabled={!canRun || runState === "running"}
          className="self-end rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium"
        >
          Analyze Longitudinal Change
        </button>
      </div>

      <p
        data-testid="lmt-status"
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
          <dl
            className="grid grid-cols-2 gap-2 text-xs"
            data-testid="lmt-summary"
          >
            <div>
              <dt className="text-gray-400">Metric</dt>
              <dd>{result.metric}</dd>
            </div>
            <div>
              <dt className="text-gray-400">Sample count</dt>
              <dd>{result.sample_count}</dd>
            </div>
            <div>
              <dt className="text-gray-400">Slope / day</dt>
              <dd>{formatStat(result.slope_per_day)}</dd>
            </div>
            <div>
              <dt className="text-gray-400">Robust slope / day</dt>
              <dd>{formatStat(result.robust_slope_per_day)}</dd>
            </div>
            <div>
              <dt className="text-gray-400">p-value</dt>
              <dd>{formatStat(result.p_value)}</dd>
            </div>
            <div>
              <dt className="text-gray-400">Earliest mean</dt>
              <dd>{formatStat(result.earliest_mean)}</dd>
            </div>
            <div>
              <dt className="text-gray-400">Latest mean</dt>
              <dd>{formatStat(result.latest_mean)}</dd>
            </div>
          </dl>

          <div data-testid="lmt-rolling-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Rolling Series
            </h3>
            <RollingTable result={result} timeColumn={timeColumn} />
          </div>

          <div data-testid="lmt-change-candidates-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Candidate Change Points
            </h3>
            <ChangeCandidatesTable result={result} />
          </div>
        </>
      )}
    </section>
  );
}

export default LaunchMonitorTrendsPanel;
