/**
 * Launch Monitor Analytics — Flexible Analysis page (#11987).
 *
 * Web counterpart of the desktop "Flexible Analysis" tab
 * (`src/tools/launch_monitor_analytics/flexible_analysis_widget.py`): load a
 * CSV in the browser, choose an outcome/predictors/analysis settings mirrored
 * from that widget's form, and run the same contract through the
 * evidence-bearing `POST /v2/analyze` endpoint (the desktop tab calls the
 * dataclass contract directly against a local DataFrame; the web client has
 * no filesystem access, so every row the backend sees is the CSV the user
 * just picked).
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ChangeEvent,
} from "react";
import { WorkspaceShell } from "@/components/layout/WorkspaceShell";
import {
  fetchLaunchMonitorAnalyticsCapabilities,
  formatStat,
  runFlexibleAnalysisV2,
  type FlexibleAnalysisResultPayload,
  type FlexibleCorrelationEstimate,
  type FlexibleRegressionEstimate,
  type LaunchMonitorAnalyticsCapabilities,
} from "@/api/launchMonitorAnalytics";
import type {
  FlexibleAnalysisPayload,
  LaunchMonitorAnalysisResultV2,
} from "@/api/generated/types";
import { LaunchMonitorTrendsPanel } from "./LaunchMonitorTrendsPanel";

const BOUNDARY_TEXT =
  "Associations and fitted regressions do not establish causality. " +
  "Aggregate observations are excluded from regression, and source-specific " +
  "fields cannot be pooled across monitor vendors.";

const DEFAULT_STATUS =
  "Load a CSV, then select an outcome and one or more predictors to run " +
  "the analysis.";

/** Contract-literal fallback (matches `GET /capabilities`) if that request fails. */
const FALLBACK_CAPABILITIES: LaunchMonitorAnalyticsCapabilities = {
  analysis_modes: ["comprehensive", "correlation", "regression"],
  correlation_methods: ["pearson", "spearman", "kendall"],
  missing_policies: ["pairwise", "listwise", "fail"],
  maximum_inline_records: 20_000,
};

const NONE_GROUP = "(none)";

export type CsvValue = string | number | null;

interface ParsedCsv {
  columns: string[];
  records: Record<string, CsvValue>[];
}

/** Minimal RFC4180 tokenizer: handles quoted fields, embedded commas/newlines. */
function tokenizeCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  let index = 0;
  while (index < text.length) {
    const char = text[index];
    if (inQuotes) {
      if (char === '"') {
        if (text[index + 1] === '"') {
          field += '"';
          index += 2;
          continue;
        }
        inQuotes = false;
        index += 1;
        continue;
      }
      field += char;
      index += 1;
      continue;
    }
    if (char === '"') {
      inQuotes = true;
      index += 1;
      continue;
    }
    if (char === ",") {
      row.push(field);
      field = "";
      index += 1;
      continue;
    }
    if (char === "\r") {
      index += 1;
      continue;
    }
    if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      index += 1;
      continue;
    }
    field += char;
    index += 1;
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

const NUMERIC_PATTERN = /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i;

/** Numeric strings become numbers, empty fields become null (issue #11987). */
function coerceCsvValue(raw: string): CsvValue {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  if (NUMERIC_PATTERN.test(trimmed)) {
    const parsed = Number(trimmed);
    if (Number.isFinite(parsed)) return parsed;
  }
  return raw;
}

/** Parse CSV text into a header-keyed record array; never touches the server. */
function parseCsv(text: string): ParsedCsv {
  const rows = tokenizeCsv(text);
  if (rows.length === 0) {
    return { columns: [], records: [] };
  }
  const [header, ...dataRows] = rows;
  const columns = header.map((cell) => cell.trim());
  const records = dataRows
    .filter((cells) => cells.length > 1 || (cells[0] ?? "").trim() !== "")
    .map((cells) => {
      const record: Record<string, CsvValue> = {};
      columns.forEach((column, columnIndex) => {
        record[column] = coerceCsvValue(cells[columnIndex] ?? "");
      });
      return record;
    });
  return { columns, records };
}

/** A column is a usable outcome/predictor once >=3 rows parse as numeric. */
export function numericColumns(
  columns: string[],
  records: Record<string, CsvValue>[],
): string[] {
  return columns.filter((column) => {
    let numericCount = 0;
    for (const record of records) {
      if (typeof record[column] === "number") numericCount += 1;
    }
    return numericCount >= 3;
  });
}

/** Grouping candidates mirror the desktop widget: any value, <=100 distinct. */
function groupColumns(
  columns: string[],
  records: Record<string, CsvValue>[],
): string[] {
  return columns.filter((column) => {
    const seen = new Set<CsvValue>();
    let hasValue = false;
    for (const record of records) {
      const value = record[column];
      if (value !== null) {
        hasValue = true;
        seen.add(value);
      }
    }
    return hasValue && seen.size <= 100;
  });
}

type RunState = "idle" | "running" | "done" | "error";

/**
 * One labelled `<select>`, reused for every single-choice control (DRY: the
 * desktop widget's mode/method/missing-policy/group-by combo boxes were the
 * same five-line block repeated four times).
 */
function SelectField<T extends string>({
  label,
  value,
  onChange,
  options,
  placeholder,
}: {
  label: string;
  value: T;
  onChange: (value: T) => void;
  options: readonly T[];
  placeholder?: { value: T; label: string };
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs uppercase tracking-wide text-gray-400">
        {label}
      </span>
      <select
        className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {placeholder && (
          <option value={placeholder.value}>{placeholder.label}</option>
        )}
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

/** One correlation/regression table pair, reused for the top-level and every group. */
function CorrelationsTable({
  correlations,
}: {
  correlations: FlexibleCorrelationEstimate[];
}) {
  if (correlations.length === 0) {
    return (
      <p className="text-xs text-gray-400">No correlations were computed.</p>
    );
  }
  return (
    <table className="w-full text-xs text-left">
      <thead>
        <tr className="text-gray-400 border-b border-gray-700">
          <th className="px-2 py-1">Predictor</th>
          <th className="px-2 py-1">r</th>
          <th className="px-2 py-1">p</th>
          <th className="px-2 py-1">adj. p</th>
          <th className="px-2 py-1">CI lower</th>
          <th className="px-2 py-1">CI upper</th>
          <th className="px-2 py-1">n</th>
        </tr>
      </thead>
      <tbody>
        {correlations.map((item) => (
          <tr
            key={item.predictor}
            className="border-b border-gray-800 text-gray-200"
          >
            <td className="px-2 py-1">
              {item.predictor}
              {item.is_boolean_projected && (
                <span
                  className="ml-1 text-amber-400"
                  title="Boolean column projected to 0/1"
                >
                  (0/1)
                </span>
              )}
            </td>
            <td className="px-2 py-1">{formatStat(item.coefficient)}</td>
            <td className="px-2 py-1">{formatStat(item.p_value)}</td>
            <td className="px-2 py-1">{formatStat(item.adjusted_p_value)}</td>
            <td className="px-2 py-1">{formatStat(item.ci_lower)}</td>
            <td className="px-2 py-1">{formatStat(item.ci_upper)}</td>
            <td className="px-2 py-1">{item.sample_count}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RegressionTable({
  regression,
}: {
  regression: FlexibleRegressionEstimate | null;
}) {
  if (!regression) {
    return (
      <p className="text-xs text-gray-400">Regression was not computed.</p>
    );
  }
  const rows = Object.entries(regression.coefficients);
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-gray-400">
        n = {regression.sample_count} · R² = {formatStat(regression.r_squared)}{" "}
        · adj. R² = {formatStat(regression.adjusted_r_squared)}
      </p>
      <table className="w-full text-xs text-left">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="px-2 py-1">Term</th>
            <th className="px-2 py-1">Estimate</th>
            <th className="px-2 py-1">SE</th>
            <th className="px-2 py-1">t</th>
            <th className="px-2 py-1">p</th>
            <th className="px-2 py-1">CI lower</th>
            <th className="px-2 py-1">CI upper</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, coefficient]) => (
            <tr key={name} className="border-b border-gray-800 text-gray-200">
              <td className="px-2 py-1">{name}</td>
              <td className="px-2 py-1">{formatStat(coefficient.estimate)}</td>
              <td className="px-2 py-1">
                {formatStat(coefficient.standard_error)}
              </td>
              <td className="px-2 py-1">
                {formatStat(coefficient.t_statistic)}
              </td>
              <td className="px-2 py-1">{formatStat(coefficient.p_value)}</td>
              <td className="px-2 py-1">{formatStat(coefficient.ci_lower)}</td>
              <td className="px-2 py-1">{formatStat(coefficient.ci_upper)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-400">
        RMSE {formatStat(regression.residual_diagnostics.rmse)} · MAE{" "}
        {formatStat(regression.residual_diagnostics.mae)} · Durbin-Watson{" "}
        {formatStat(regression.residual_diagnostics.durbin_watson)} ·
        influential points {regression.residual_diagnostics.influential_count}
      </p>
    </div>
  );
}

export function LaunchMonitorAnalyticsPage() {
  const [capabilities, setCapabilities] =
    useState<LaunchMonitorAnalyticsCapabilities>(FALLBACK_CAPABILITIES);

  const [fileName, setFileName] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [records, setRecords] = useState<Record<string, CsvValue>[]>([]);

  const [outcome, setOutcome] = useState("");
  const [predictors, setPredictors] = useState<string[]>([]);
  const [analysisMode, setAnalysisMode] =
    useState<FlexibleAnalysisPayload["analysis_mode"]>("comprehensive");
  const [correlationMethod, setCorrelationMethod] =
    useState<FlexibleAnalysisPayload["correlation_method"]>("pearson");
  const [missingPolicy, setMissingPolicy] =
    useState<FlexibleAnalysisPayload["missing_policy"]>("pairwise");
  const [groupBy, setGroupBy] = useState(NONE_GROUP);
  const [minSamples, setMinSamples] = useState(10);

  const [runState, setRunState] = useState<RunState>("idle");
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<LaunchMonitorAnalysisResultV2 | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;
    void fetchLaunchMonitorAnalyticsCapabilities()
      .then((data) => {
        if (!cancelled) setCapabilities(data);
      })
      .catch(() => {
        // Keep the contract-literal fallback; the form stays usable.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const numericCols = useMemo(
    () => numericColumns(columns, records),
    [columns, records],
  );
  const groupCols = useMemo(
    () => groupColumns(columns, records),
    [columns, records],
  );
  const predictorOptions = useMemo(
    () => numericCols.filter((column) => column !== outcome),
    [numericCols, outcome],
  );

  // Drop selections a newly loaded CSV (or outcome change) no longer supports.
  useEffect(() => {
    setOutcome((prev) => (numericCols.includes(prev) ? prev : ""));
  }, [numericCols]);
  useEffect(() => {
    setPredictors((prev) =>
      prev.filter((p) => p !== outcome && numericCols.includes(p)),
    );
  }, [outcome, numericCols]);
  useEffect(() => {
    setGroupBy((prev) =>
      prev === NONE_GROUP || groupCols.includes(prev) ? prev : NONE_GROUP,
    );
  }, [groupCols]);

  const loadCsvFile = useCallback(async (file: File) => {
    try {
      const text = await file.text();
      const parsed = parseCsv(text);
      if (parsed.columns.length === 0 || parsed.records.length === 0) {
        throw new Error(`"${file.name}" has no parseable rows`);
      }
      setColumns(parsed.columns);
      setRecords(parsed.records);
      setFileName(file.name);
      setParseError(null);
      setResult(null);
      setRunState("idle");
      setRunError(null);
    } catch (err) {
      setParseError(
        err instanceof Error ? err.message : `Failed to read "${file.name}"`,
      );
    }
  }, []);

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      // Allow re-selecting the same file after a parse failure.
      event.target.value = "";
      if (file) void loadCsvFile(file);
    },
    [loadCsvFile],
  );

  const canRun =
    outcome !== "" && predictors.length > 0 && records.length >= 3;

  const handleRun = useCallback(() => {
    if (!canRun) return;
    setRunState("running");
    setRunError(null);
    const analysis: FlexibleAnalysisPayload = {
      outcome,
      predictors,
      analysis_mode: analysisMode,
      correlation_method: correlationMethod,
      missing_policy: missingPolicy,
      group_by: groupBy === NONE_GROUP ? null : groupBy,
      confidence_level: 0.95,
      min_samples: minSamples,
      allow_aggregate: false,
    };
    void runFlexibleAnalysisV2(records, analysis)
      .then((data) => {
        setResult(data);
        setRunState("done");
      })
      .catch((err) => {
        setRunError(err instanceof Error ? err.message : "Analysis failed");
        setRunState("error");
      });
  }, [
    canRun,
    outcome,
    predictors,
    analysisMode,
    correlationMethod,
    missingPolicy,
    groupBy,
    minSamples,
    records,
  ]);

  const analysisPayload = (result?.analysis ?? null) as
    | FlexibleAnalysisResultPayload
    | null;

  const warnings = useMemo(() => {
    const combined = [
      ...(result?.warnings ?? []),
      ...(analysisPayload?.warnings ?? []),
    ];
    return Array.from(new Set(combined));
  }, [result, analysisPayload]);

  const unavailableReasons = useMemo(() => {
    if (!result) return [];
    return result.availability
      .filter((item) => item.state === "unavailable")
      .map((item) => item.message)
      .filter((message): message is string => Boolean(message));
  }, [result]);

  const statusMessage = useMemo(() => {
    if (runState === "running") return "Running analysis…";
    if (runState === "error") {
      return runError
        ? `Analysis could not run: ${runError}`
        : "Analysis could not run.";
    }
    if (runState === "idle" || !result) return DEFAULT_STATUS;
    const rowCount =
      analysisPayload?.dataset.row_count ?? result.missingness.complete_row_count;
    if (result.status === "available") {
      return `Analysis complete for ${rowCount} observation(s).`;
    }
    if (result.status === "partial") {
      return (
        `Analysis complete for ${rowCount} observation(s) with partial ` +
        "availability — see notes below."
      );
    }
    return unavailableReasons.length > 0
      ? `Analysis could not run: ${unavailableReasons.join("; ")}`
      : "Analysis is unavailable for the selected variables.";
  }, [runState, runError, result, analysisPayload, unavailableReasons]);

  const leftPanel = (
    <div className="flex flex-col gap-3 p-4 text-sm text-gray-200">
      <div>
        <h1 className="text-lg font-semibold text-white">
          Launch Monitor Analytics
        </h1>
        <p className="text-xs text-gray-400 mt-1">Flexible Analysis</p>
      </div>

      <p
        className="text-xs text-amber-200/90 border border-amber-700/40 rounded p-2"
        aria-label="Flexible Analysis Scientific Boundary"
      >
        {BOUNDARY_TEXT}
      </p>

      <label className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-wide text-gray-400">
          Load CSV
        </span>
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={handleFileChange}
          className="text-xs"
          aria-label="Load CSV"
          data-testid="csv-file-input"
        />
      </label>
      {fileName && !parseError && (
        <p className="text-xs text-gray-400" data-testid="lma-row-col-count">
          {fileName}: {records.length} row(s) · {columns.length} column(s)
        </p>
      )}
      {parseError && (
        <p className="text-xs text-red-300" role="alert">
          {parseError}
        </p>
      )}

      <SelectField
        label="Outcome"
        value={outcome}
        onChange={setOutcome}
        options={numericCols}
        placeholder={{ value: "", label: "(select outcome)" }}
      />

      <label className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-wide text-gray-400">
          Predictors
        </span>
        <select
          multiple
          size={6}
          className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
          value={predictors}
          onChange={(e) =>
            setPredictors(
              Array.from(e.target.selectedOptions).map((o) => o.value),
            )
          }
        >
          {predictorOptions.map((column) => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </select>
      </label>

      <SelectField
        label="Analysis Mode"
        value={analysisMode}
        onChange={setAnalysisMode}
        options={capabilities.analysis_modes}
      />

      <SelectField
        label="Correlation Method"
        value={correlationMethod}
        onChange={setCorrelationMethod}
        options={capabilities.correlation_methods}
      />

      <SelectField
        label="Missing-Data Policy"
        value={missingPolicy}
        onChange={setMissingPolicy}
        options={capabilities.missing_policies}
      />

      <SelectField
        label="Group By"
        value={groupBy}
        onChange={setGroupBy}
        options={groupCols}
        placeholder={{ value: NONE_GROUP, label: NONE_GROUP }}
      />

      <label className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-wide text-gray-400">
          Minimum Samples
        </span>
        <input
          type="number"
          min={3}
          value={minSamples}
          onChange={(e) =>
            setMinSamples(Math.max(3, Number(e.target.value) || 3))
          }
          className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
        />
      </label>

      <button
        type="button"
        onClick={handleRun}
        disabled={!canRun || runState === "running"}
        className="rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium"
      >
        Run Flexible Analysis
      </button>
    </div>
  );

  const mainContent = (
    <div className="flex flex-col h-full gap-4 p-4 overflow-y-auto text-sm text-gray-200">
      <p
        data-testid="lma-status"
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

      {warnings.length > 0 && (
        <ul className="text-xs text-amber-300 list-disc list-inside">
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      {result && unavailableReasons.length > 0 && result.status !== "unavailable" && (
        <ul className="text-xs text-amber-300 list-disc list-inside">
          {unavailableReasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}

      {analysisPayload && (
        <>
          <section className="rounded border border-gray-700 bg-gray-800 p-3">
            <h3 className="text-sm font-medium text-white mb-2">Dataset</h3>
            <dl className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <dt className="text-gray-400">Rows</dt>
                <dd>{analysisPayload.dataset.row_count}</dd>
              </div>
              <div>
                <dt className="text-gray-400">Complete rows</dt>
                <dd>{analysisPayload.dataset.complete_row_count}</dd>
              </div>
              <div>
                <dt className="text-gray-400">Monitor vendors</dt>
                <dd>{analysisPayload.dataset.monitor_vendors.join(", ") || "—"}</dd>
              </div>
              <div>
                <dt className="text-gray-400">Observation kinds</dt>
                <dd>{analysisPayload.dataset.observation_kinds.join(", ") || "—"}</dd>
              </div>
            </dl>
          </section>

          <section
            className="rounded border border-gray-700 bg-gray-800 p-3"
            data-testid="lma-correlations-table"
          >
            <h3 className="text-sm font-medium text-white mb-2">Correlations</h3>
            <CorrelationsTable correlations={analysisPayload.correlations} />
          </section>

          <section
            className="rounded border border-gray-700 bg-gray-800 p-3"
            data-testid="lma-regression-table"
          >
            <h3 className="text-sm font-medium text-white mb-2">
              Regression
            </h3>
            <RegressionTable regression={analysisPayload.regression} />
          </section>

          {analysisPayload.groups.length > 0 && (
            <section className="rounded border border-gray-700 bg-gray-800 p-3">
              <h3 className="text-sm font-medium text-white mb-2">
                Group Analyses — {groupBy}
              </h3>
              <div className="flex flex-col gap-3">
                {analysisPayload.groups.map((group) => (
                  <div
                    key={group.group_value}
                    className="border-t border-gray-700 pt-2 first:border-t-0 first:pt-0"
                  >
                    <p className="text-xs text-gray-400 mb-1">
                      {group.group_value} (n = {group.row_count})
                    </p>
                    <CorrelationsTable correlations={group.correlations} />
                    <RegressionTable regression={group.regression} />
                    {group.warnings.map((warning) => (
                      <p key={warning} className="text-xs text-amber-300 mt-1">
                        {warning}
                      </p>
                    ))}
                  </div>
                ))}
              </div>
            </section>
          )}

          <details className="rounded border border-gray-700 bg-gray-800 p-3">
            <summary className="text-sm font-medium text-white cursor-pointer">
              Traceable Details
            </summary>
            <pre className="text-xs text-gray-400 whitespace-pre-wrap mt-2">
              {JSON.stringify(analysisPayload, null, 2)}
            </pre>
          </details>
        </>
      )}

      <LaunchMonitorTrendsPanel columns={columns} records={records} />
    </div>
  );

  return (
    <WorkspaceShell leftPanel={leftPanel}>
      <main id="main-content" className="min-h-0 min-w-0 flex-1">
        {mainContent}
      </main>
    </WorkspaceShell>
  );
}

export default LaunchMonitorAnalyticsPage;
