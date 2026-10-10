/**
 * Launch Monitor Analytics — Data Treatment panel (#11987, slice 6b).
 *
 * Web counterpart of the desktop "Data Treatment" tab
 * (`src/tools/launch_monitor_analytics/gui.py` `_build_treatment_tab` /
 * `_read_treatment_config` / `_filter_rules` / `_compute_treatment`): name
 * required/robust-outlier metrics, a modified-z threshold, an
 * exclude-flagged switch and a table of structured subset filters, then run
 * the same `apply_treatment` contract through `POST /v2/treatment`. Unlike
 * every sibling panel, this one is applied to the raw CSV (`records`), not
 * the page's already-treated analysis view — mirroring how the desktop tab
 * treats `project.combined_shots()`, never a prior treatment's output.
 */

import { useCallback, useMemo, useState } from "react";
import {
  applyTreatmentV2,
  type FilterOperator,
  type TreatmentResponse,
} from "@/api/launchMonitorAnalytics";
import type { FilterRulePayload } from "@/api/generated/types";
import type { CsvValue } from "./LaunchMonitorAnalytics";

/** Same eight choices as the desktop filter table's operator combo. */
const OPERATOR_OPTIONS: FilterOperator[] = [
  "eq",
  "ne",
  "lt",
  "le",
  "gt",
  "ge",
  "contains",
  "in",
];
/** Matches `_read_treatment_config`'s `self.outlier_threshold_spin` default. */
const DEFAULT_THRESHOLD = 4.5;
const MIN_THRESHOLD = 1;
const MAX_THRESHOLD = 20;

const BUTTON_CLASS =
  "self-end rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium";
const SECONDARY_BUTTON_CLASS =
  "self-end rounded bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium";

const DEFAULT_STATUS =
  "Name required/outlier metrics and filters, then apply the treatment.";

type RunState = "idle" | "running" | "done" | "error";

/** One structured subset filter row, local-only until Apply is pressed. */
interface FilterRow {
  id: number;
  column: string;
  operator: FilterOperator;
  value: string;
}

/** Comma-separated text field to a trimmed, blank-dropping list (desktop parity: `_read_treatment_config`). */
function parseMetricList(text: string): string[] {
  return text
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item !== "");
}

function FlagsTable({ flags }: { flags: TreatmentResponse["flags"] }) {
  if (flags.length === 0) {
    return <p className="text-xs text-gray-400">No rows were flagged.</p>;
  }
  return (
    <table className="w-full text-xs text-left">
      <thead>
        <tr className="text-gray-400 border-b border-gray-700">
          <th className="px-2 py-1">Row</th>
          <th className="px-2 py-1">Flag Type</th>
          <th className="px-2 py-1">Metric</th>
        </tr>
      </thead>
      <tbody>
        {flags.map((flag, index) => (
          <tr
            key={`${flag.row_index}-${flag.flag_type}-${index}`}
            className="border-b border-gray-800 text-gray-200"
          >
            <td className="px-2 py-1">{String(flag.row_index)}</td>
            <td className="px-2 py-1">{flag.flag_type}</td>
            <td className="px-2 py-1">{flag.metric ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function LaunchMonitorTreatmentPanel({
  columns,
  records,
  onTreated,
}: {
  columns: string[];
  records: Record<string, CsvValue>[];
  onTreated: (data: Record<string, CsvValue>[] | null) => void;
}) {
  const [requiredMetricsText, setRequiredMetricsText] = useState("");
  const [outlierMetricsText, setOutlierMetricsText] = useState("");
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [excludeFlagged, setExcludeFlagged] = useState(false);
  const [filterRows, setFilterRows] = useState<FilterRow[]>([]);
  const [nextRowId, setNextRowId] = useState(0);

  const [runState, setRunState] = useState<RunState>("idle");
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<TreatmentResponse | null>(null);

  const canApply = records.length > 0;

  const handleAddFilter = useCallback(() => {
    setFilterRows((rows) => [
      ...rows,
      { id: nextRowId, column: columns[0] ?? "", operator: "eq", value: "" },
    ]);
    setNextRowId((id) => id + 1);
  }, [columns, nextRowId]);

  const handleRemoveFilter = useCallback((id: number) => {
    setFilterRows((rows) => rows.filter((row) => row.id !== id));
  }, []);

  const handleFilterChange = useCallback(
    (id: number, patch: Partial<Omit<FilterRow, "id">>) => {
      setFilterRows((rows) =>
        rows.map((row) => (row.id === id ? { ...row, ...patch } : row)),
      );
    },
    [],
  );

  const handleThresholdChange = useCallback((raw: string) => {
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return;
    setThreshold(Math.min(MAX_THRESHOLD, Math.max(MIN_THRESHOLD, parsed)));
  }, []);

  const handleApply = useCallback(() => {
    if (!canApply) return;
    setRunState("running");
    setRunError(null);
    const filters: FilterRulePayload[] = filterRows.map((row) => ({
      column: row.column,
      operator: row.operator,
      value: row.value,
    }));
    void applyTreatmentV2(records, {
      requiredMetrics: parseMetricList(requiredMetricsText),
      outlierMetrics: parseMetricList(outlierMetricsText),
      robustZThreshold: threshold,
      excludeFlagged,
      filters,
    })
      .then((data) => {
        setResult(data);
        setRunState("done");
        onTreated(data.data);
      })
      .catch((err) => {
        setRunError(
          err instanceof Error ? err.message : "Data treatment failed",
        );
        setRunState("error");
      });
  }, [
    canApply,
    records,
    requiredMetricsText,
    outlierMetricsText,
    threshold,
    excludeFlagged,
    filterRows,
    onTreated,
  ]);

  const handleReset = useCallback(() => {
    setResult(null);
    setRunState("idle");
    setRunError(null);
    onTreated(null);
  }, [onTreated]);

  const statusMessage = useMemo(() => {
    if (runState === "running") return "Applying reproducible treatment…";
    if (runState === "error") {
      return runError
        ? `Data treatment could not run: ${runError}`
        : "Data treatment could not run.";
    }
    if (runState === "done" && result) {
      return (
        `${result.flag_count} flags; ${result.shot_count} shots in the ` +
        "analysis view. Raw sessions are unchanged."
      );
    }
    return DEFAULT_STATUS;
  }, [runState, runError, result]);

  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 flex flex-col gap-3">
      <h2 className="text-sm font-medium text-white">Data Treatment</h2>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Required Metrics
          </span>
          <input
            type="text"
            placeholder="club_speed, ball_speed"
            value={requiredMetricsText}
            onChange={(e) => setRequiredMetricsText(e.target.value)}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Robust-Outlier Metrics
          </span>
          <input
            type="text"
            placeholder="club_speed, ball_speed"
            value={outlierMetricsText}
            onChange={(e) => setOutlierMetricsText(e.target.value)}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Modified Z Threshold
          </span>
          <input
            type="number"
            min={MIN_THRESHOLD}
            max={MAX_THRESHOLD}
            step={0.25}
            value={threshold}
            onChange={(e) => handleThresholdChange(e.target.value)}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          />
        </label>

        <label className="flex flex-row items-center gap-2 self-end pb-2">
          <input
            type="checkbox"
            checked={excludeFlagged}
            onChange={(e) => setExcludeFlagged(e.target.checked)}
          />
          <span className="text-xs text-gray-300">
            Exclude Flagged Rows from Analysis View
          </span>
        </label>

        <button
          type="button"
          onClick={handleApply}
          disabled={!canApply || runState === "running"}
          className={BUTTON_CLASS}
        >
          Apply Reproducible Treatment
        </button>
        <button type="button" onClick={handleReset} className={SECONDARY_BUTTON_CLASS}>
          Reset Treatment
        </button>
      </div>

      <div className="flex flex-col gap-2" data-testid="lmt-filter-rules">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-medium text-white">
            Structured Subset Filters
          </h3>
          <button
            type="button"
            onClick={handleAddFilter}
            className="rounded bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 text-xs"
          >
            Add Filter
          </button>
        </div>
        {filterRows.length === 0 ? (
          <p className="text-xs text-gray-400">No filters defined.</p>
        ) : (
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="px-2 py-1">Column</th>
                <th className="px-2 py-1">Operator</th>
                <th className="px-2 py-1">Value</th>
                <th className="px-2 py-1" />
              </tr>
            </thead>
            <tbody>
              {filterRows.map((row) => (
                <tr key={row.id} className="border-b border-gray-800">
                  <td className="px-2 py-1">
                    <select
                      aria-label="Filter Column"
                      value={row.column}
                      onChange={(e) =>
                        handleFilterChange(row.id, { column: e.target.value })
                      }
                      className="rounded bg-gray-900 border border-gray-700 px-1 py-0.5"
                    >
                      {columns.map((column) => (
                        <option key={column} value={column}>
                          {column}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="px-2 py-1">
                    <select
                      aria-label="Filter Operator"
                      value={row.operator}
                      onChange={(e) =>
                        handleFilterChange(row.id, {
                          operator: e.target.value as FilterOperator,
                        })
                      }
                      className="rounded bg-gray-900 border border-gray-700 px-1 py-0.5"
                    >
                      {OPERATOR_OPTIONS.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="px-2 py-1">
                    <input
                      aria-label="Filter Value"
                      type="text"
                      value={row.value}
                      onChange={(e) =>
                        handleFilterChange(row.id, { value: e.target.value })
                      }
                      className="rounded bg-gray-900 border border-gray-700 px-1 py-0.5"
                    />
                  </td>
                  <td className="px-2 py-1">
                    <button
                      type="button"
                      onClick={() => handleRemoveFilter(row.id)}
                      className="text-xs text-red-300 hover:text-red-200"
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
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
          <div data-testid="lmt-flags-table">
            <h3 className="text-xs font-medium text-white mb-1">Flags</h3>
            <div className="overflow-x-auto">
              <FlagsTable flags={result.flags} />
            </div>
          </div>

          <details data-testid="lmt-audit-log">
            <summary className="text-xs font-medium text-white cursor-pointer">
              Audit Log
            </summary>
            <pre className="text-xs text-gray-400 whitespace-pre-wrap mt-2">
              {JSON.stringify(result.audit_log, null, 2)}
            </pre>
          </details>
        </>
      )}
    </section>
  );
}

export default LaunchMonitorTreatmentPanel;
