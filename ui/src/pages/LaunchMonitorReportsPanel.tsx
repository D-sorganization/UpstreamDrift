/**
 * Launch Monitor Analytics — Reports panel (#11987, slice 7b).
 *
 * Web counterpart of the desktop "Reports" tab
 * (`src/tools/launch_monitor_analytics/gui.py` `_refresh_report`/
 * `export_data`/`export_manifest`): name the project, build the report text,
 * and export the canonical CSV plus reproducibility manifest, via the same
 * `POST /v2/report` / `POST /v2/export` contract the desktop tab's builders
 * back. Shares the CSV (and Data Treatment's audit log) already loaded by
 * `LaunchMonitorAnalyticsPage` — no separate upload step.
 */

import { useCallback, useState } from "react";
import {
  buildReportV2,
  exportReportV2,
  type ExportReportResponse,
  type ReportResponse,
} from "@/api/launchMonitorAnalytics";
import type { CsvValue } from "./LaunchMonitorAnalytics";

const DEFAULT_PROJECT_NAME = "Untitled Launch Monitor Study";

const DEFAULT_STATUS =
  "Load a CSV, then generate a report or export the canonical CSV and " +
  "reproducibility manifest.";

const RUN_BUTTON_CLASS =
  "self-end rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium";

type RunState = "idle" | "running" | "done" | "error";

export function LaunchMonitorReportsPanel({
  records,
  treatmentAuditLog = [],
}: {
  records: Record<string, CsvValue>[];
  treatmentAuditLog?: Record<string, unknown>[];
}) {
  const [projectName, setProjectName] = useState(DEFAULT_PROJECT_NAME);

  const [reportState, setReportState] = useState<RunState>("idle");
  const [reportError, setReportError] = useState<string | null>(null);
  const [report, setReport] = useState<ReportResponse | null>(null);

  const [exportState, setExportState] = useState<RunState>("idle");
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportResult, setExportResult] = useState<ExportReportResponse | null>(
    null,
  );

  const isRunning = reportState === "running" || exportState === "running";
  const canRun = records.length > 0 && !isRunning;

  const handleGenerateReport = useCallback(() => {
    if (records.length === 0) return;
    setReportState("running");
    setReportError(null);
    void buildReportV2(records, projectName, treatmentAuditLog)
      .then((data) => {
        setReport(data);
        setReportState("done");
      })
      .catch((err) => {
        setReportError(
          err instanceof Error ? err.message : "Report generation failed",
        );
        setReportState("error");
      });
  }, [records, projectName, treatmentAuditLog]);

  const handleExport = useCallback(() => {
    if (records.length === 0) return;
    setExportState("running");
    setExportError(null);
    void exportReportV2(records, projectName, treatmentAuditLog)
      .then((data) => {
        setExportResult(data);
        setExportState("done");
      })
      .catch((err) => {
        setExportError(err instanceof Error ? err.message : "Export failed");
        setExportState("error");
      });
  }, [records, projectName, treatmentAuditLog]);

  const statusMessage = (() => {
    if (reportState === "running" || exportState === "running") {
      return "Working…";
    }
    if (reportState === "error") {
      return reportError
        ? `Report could not be generated: ${reportError}`
        : "Report could not be generated.";
    }
    if (exportState === "error") {
      return exportError
        ? `Export could not run: ${exportError}`
        : "Export could not run.";
    }
    if (reportState === "done" || exportState === "done") {
      return "Done.";
    }
    return DEFAULT_STATUS;
  })();

  const statusClass =
    reportState === "error" || exportState === "error"
      ? "text-red-300"
      : reportState === "done" || exportState === "done"
        ? "text-emerald-300"
        : "text-gray-300";

  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 flex flex-col gap-3">
      <h2 className="text-sm font-medium text-white">Reports</h2>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Project Name
          </span>
          <input
            type="text"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            data-testid="lmr-project-name"
          />
        </label>

        <button
          type="button"
          onClick={handleGenerateReport}
          disabled={!canRun}
          className={RUN_BUTTON_CLASS}
          data-testid="lmr-generate-btn"
        >
          Generate Report
        </button>

        <button
          type="button"
          onClick={handleExport}
          disabled={!canRun}
          className={RUN_BUTTON_CLASS}
          data-testid="lmr-export-btn"
        >
          Export Canonical CSV & Manifest
        </button>
      </div>

      <p data-testid="lmr-status" className={statusClass}>
        {statusMessage}
      </p>

      {report && (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-gray-400" data-testid="lmr-report-summary">
            {report.project_name}: {report.shot_count} shot(s) ·{" "}
            {report.session_count} session(s) ·{" "}
            {report.canonical_metrics.join(", ")}
          </p>
          <pre
            data-testid="lmr-report-text"
            className="text-xs text-gray-200 whitespace-pre-wrap bg-gray-900 rounded p-2"
          >
            {report.report_text}
          </pre>
        </div>
      )}

      {exportResult && (
        <div className="flex flex-col gap-2">
          <div>
            <h3 className="text-xs font-medium text-white mb-1">
              Canonical CSV
            </h3>
            <pre
              data-testid="lmr-export-csv"
              className="text-xs text-gray-200 whitespace-pre-wrap bg-gray-900 rounded p-2 max-h-48 overflow-y-auto"
            >
              {exportResult.csv}
            </pre>
          </div>
          <div>
            <h3 className="text-xs font-medium text-white mb-1">
              Reproducibility Manifest
            </h3>
            <pre
              data-testid="lmr-export-manifest"
              className="text-xs text-gray-200 whitespace-pre-wrap bg-gray-900 rounded p-2 max-h-48 overflow-y-auto"
            >
              {JSON.stringify(exportResult.manifest, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </section>
  );
}

export default LaunchMonitorReportsPanel;
