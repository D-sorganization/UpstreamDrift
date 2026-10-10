/**
 * Launch Monitor Analytics — Dispersion panel (#11987, slice 3b).
 *
 * Web counterpart of the desktop "Dispersion" tab
 * (`src/tools/launch_monitor_analytics/gui.py` `_build_dispersion_tab` /
 * `_read_dispersion_params` / `_compute_dispersion`): pick forward/lateral
 * coordinate columns and an optional group-by column, then run the same
 * `analyze_dispersion` contract through `POST /v2/dispersion`. Shares the CSV
 * already loaded by `LaunchMonitorAnalyticsPage` — no separate upload step.
 */

import { useCallback, useMemo, useState } from "react";
import {
  analyzeDispersionV2,
  type DispersionGroupColumn,
  type DispersionGroupResult,
  type DispersionResponse,
} from "@/api/launchMonitorAnalytics";
import { numericColumns, type CsvValue } from "./LaunchMonitorAnalytics";

/** Matches `_read_dispersion_params`'s `self.dispersion_forward` default. */
const DEFAULT_FORWARD = "carry_distance";
/** Matches `_read_dispersion_params`'s `self.dispersion_lateral` default. */
const DEFAULT_LATERAL = "lateral_carry";
/** Sentinel for the desktop combo's "(all shots)" item — no grouping. */
const ALL_SHOTS = "(all shots)";
/**
 * Group-by candidates, matching the desktop combo's fixed item list
 * (`_build_dispersion_tab`) and `DispersionPayloadV2.group_column`'s literal
 * union. Unlike the desktop combo (always all three), the web page only
 * offers a candidate actually present in the loaded CSV — picking an absent
 * column would silently fall back to "All Shots" on the backend anyway
 * (`analyze_dispersion_v2`).
 */
const GROUP_CANDIDATES: DispersionGroupColumn[] = [
  "monitor_vendor",
  "session_id",
  "club",
];

const DEFAULT_STATUS =
  "Select forward/lateral coordinates, then run the dispersion analysis.";

/** Points sampled around each 95% dispersion ellipse (smooth, cheap curve). */
const ELLIPSE_SAMPLE_POINTS = 48;

type RunState = "idle" | "running" | "done" | "error";

/** Render a nullable dispersion statistic — never a misleading zero. */
function formatValue(value: number | null | undefined, digits = 4): string {
  if (value == null || Number.isNaN(value)) {
    return "unavailable";
  }
  return value.toFixed(digits);
}

/** One ellipse boundary point in (forward, lateral) data coordinates. */
interface EllipsePoint {
  forward: number;
  lateral: number;
}

/**
 * Sample the 95% covariance ellipse boundary, mirroring
 * `_dispersion_plot_data` (gui.py) exactly: the same local ellipse
 * parametrization and rotation matrix, centered on the group's mean.
 * Returns `null` when any required field is missing or non-finite (a
 * JSON-safe `null` from `_json_safe_float`), rather than drawing a
 * misleading degenerate ellipse.
 */
function ellipsePoints(result: DispersionGroupResult): EllipsePoint[] | null {
  const {
    ellipse_major,
    ellipse_minor,
    ellipse_angle_rad,
    mean_forward,
    mean_lateral,
  } = result;
  if (
    ellipse_major == null ||
    !Number.isFinite(ellipse_major) ||
    ellipse_minor == null ||
    !Number.isFinite(ellipse_minor) ||
    ellipse_angle_rad == null ||
    !Number.isFinite(ellipse_angle_rad) ||
    mean_forward == null ||
    !Number.isFinite(mean_forward) ||
    mean_lateral == null ||
    !Number.isFinite(mean_lateral)
  ) {
    return null;
  }
  const cos = Math.cos(ellipse_angle_rad);
  const sin = Math.sin(ellipse_angle_rad);
  const points: EllipsePoint[] = [];
  for (let index = 0; index < ELLIPSE_SAMPLE_POINTS; index += 1) {
    const t = (2 * Math.PI * index) / ELLIPSE_SAMPLE_POINTS;
    const localForward = (ellipse_major / 2) * Math.cos(t);
    const localLateral = (ellipse_minor / 2) * Math.sin(t);
    // Same rotation as `_dispersion_plot_data`: `ellipse @ rotation.T`.
    const rotatedForward = localForward * cos - localLateral * sin;
    const rotatedLateral = localForward * sin + localLateral * cos;
    points.push({
      forward: rotatedForward + mean_forward,
      lateral: rotatedLateral + mean_lateral,
    });
  }
  return points;
}

/**
 * One group's 95% dispersion ellipse as a small inline SVG, axes labelled
 * lateral (x) / forward (y). No charting dependency — a plain scatter-free
 * polygon through the sampled boundary, centered on the group's mean.
 */
function EllipsePlot({ result }: { result: DispersionGroupResult }) {
  const points = ellipsePoints(result);
  if (!points) {
    return (
      <p
        className="text-xs text-gray-400"
        data-testid={`lmd-ellipse-unavailable-${result.group}`}
      >
        Ellipse unavailable.
      </p>
    );
  }
  const meanForward = result.mean_forward ?? 0;
  const meanLateral = result.mean_lateral ?? 0;
  const halfExtent = Math.max(result.ellipse_major ?? 0, result.ellipse_minor ?? 0) / 2;
  const radius = halfExtent > 0 ? halfExtent * 1.25 : 1;
  const path =
    points
      .map((point, index) => {
        const x = point.lateral - meanLateral;
        // Flip forward into SVG's downward-y so increasing forward distance
        // draws upward, matching a top-down dispersion-pattern convention.
        const y = -(point.forward - meanForward);
        return `${index === 0 ? "M" : "L"}${x.toFixed(3)},${y.toFixed(3)}`;
      })
      .join(" ") + " Z";
  return (
    <svg
      viewBox={`${-radius} ${-radius} ${2 * radius} ${2 * radius}`}
      width={96}
      height={96}
      className="bg-gray-900 rounded border border-gray-700"
      role="img"
      aria-label={`95% dispersion ellipse for ${result.group}`}
      data-testid={`lmd-ellipse-svg-${result.group}`}
    >
      <line
        x1={-radius}
        y1={0}
        x2={radius}
        y2={0}
        stroke="#4b5563"
        strokeWidth={0.5}
      />
      <line
        x1={0}
        y1={-radius}
        x2={0}
        y2={radius}
        stroke="#4b5563"
        strokeWidth={0.5}
      />
      <path
        d={path}
        fill="rgba(59,130,246,0.25)"
        stroke="#60a5fa"
        strokeWidth={1}
      />
      <text
        x={radius * 0.55}
        y={-radius * 0.05}
        fontSize={radius * 0.12}
        fill="#9ca3af"
      >
        Lateral
      </text>
      <text
        x={radius * 0.05}
        y={-radius * 0.6}
        fontSize={radius * 0.12}
        fill="#9ca3af"
      >
        Forward
      </text>
    </svg>
  );
}

function ResultsTable({ groups }: { groups: DispersionGroupResult[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs text-left">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="px-2 py-1">Group</th>
            <th className="px-2 py-1">n</th>
            <th className="px-2 py-1">Center Fwd</th>
            <th className="px-2 py-1">Center Lat</th>
            <th className="px-2 py-1">Mean Fwd</th>
            <th className="px-2 py-1">Mean Lat</th>
            <th className="px-2 py-1">Ellipse Major</th>
            <th className="px-2 py-1">Ellipse Minor</th>
            <th className="px-2 py-1">Angle (rad)</th>
            <th className="px-2 py-1">Area 95%</th>
            <th className="px-2 py-1">Radial RMSE</th>
            <th className="px-2 py-1">Radial P50</th>
            <th className="px-2 py-1">Radial P90</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((group) => (
            <tr
              key={group.group}
              className="border-b border-gray-800 text-gray-200"
            >
              <td className="px-2 py-1">{group.group}</td>
              <td className="px-2 py-1">{group.sample_count}</td>
              <td className="px-2 py-1">{formatValue(group.center_forward)}</td>
              <td className="px-2 py-1">{formatValue(group.center_lateral)}</td>
              <td className="px-2 py-1">{formatValue(group.mean_forward)}</td>
              <td className="px-2 py-1">{formatValue(group.mean_lateral)}</td>
              <td className="px-2 py-1">{formatValue(group.ellipse_major)}</td>
              <td className="px-2 py-1">{formatValue(group.ellipse_minor)}</td>
              <td className="px-2 py-1">
                {formatValue(group.ellipse_angle_rad)}
              </td>
              <td className="px-2 py-1">{formatValue(group.area_95)}</td>
              <td className="px-2 py-1">{formatValue(group.radial_rmse)}</td>
              <td className="px-2 py-1">{formatValue(group.radial_p50)}</td>
              <td className="px-2 py-1">{formatValue(group.radial_p90)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * The coordinate column to use: the user's choice while it is still a column
 * (or the explicit empty placeholder), otherwise the desktop tab's default,
 * otherwise the first numeric column (mirrors `_set_preferred_combo`'s no-op
 * when its preferred value is absent, which leaves the combo at its first
 * item).
 */
function effectiveCoordinate(
  choice: string | null,
  options: string[],
  preferred: string,
): string {
  if (choice !== null && (choice === "" || options.includes(choice))) {
    return choice;
  }
  if (options.includes(preferred)) return preferred;
  return options[0] ?? "";
}

export function LaunchMonitorDispersionPanel({
  columns,
  records,
}: {
  columns: string[];
  records: Record<string, CsvValue>[];
}) {
  const coordinateOptions = useMemo(
    () => numericColumns(columns, records),
    [columns, records],
  );
  const groupOptions = useMemo(
    () => GROUP_CANDIDATES.filter((candidate) => columns.includes(candidate)),
    [columns],
  );

  // The user's explicit choices; null means "not chosen yet".
  const [forwardChoice, setForward] = useState<string | null>(null);
  const [lateralChoice, setLateral] = useState<string | null>(null);
  const [groupChoice, setGroupColumn] = useState<string>(ALL_SHOTS);

  const [runState, setRunState] = useState<RunState>("idle");
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<DispersionResponse | null>(null);

  // Derived during render rather than synced in effects, so a column that
  // leaves the CSV falls back without a cascading re-render.
  const forward = effectiveCoordinate(
    forwardChoice,
    coordinateOptions,
    DEFAULT_FORWARD,
  );
  const lateral = effectiveCoordinate(
    lateralChoice,
    coordinateOptions,
    DEFAULT_LATERAL,
  );
  const groupColumn =
    groupChoice === ALL_SHOTS ||
    groupOptions.includes(groupChoice as DispersionGroupColumn)
      ? groupChoice
      : ALL_SHOTS;

  const canRun = forward !== "" && lateral !== "" && records.length >= 3;

  const handleRun = useCallback(() => {
    if (!canRun) return;
    setRunState("running");
    setRunError(null);
    const requestedGroup =
      groupColumn === ALL_SHOTS ? null : (groupColumn as DispersionGroupColumn);
    void analyzeDispersionV2(records, forward, lateral, requestedGroup)
      .then((data) => {
        setResult(data);
        setRunState("done");
      })
      .catch((err) => {
        setRunError(
          err instanceof Error ? err.message : "Dispersion analysis failed",
        );
        setRunState("error");
      });
  }, [canRun, records, forward, lateral, groupColumn]);

  const statusMessage = useMemo(() => {
    if (runState === "running") return "Running dispersion analysis…";
    if (runState === "error") {
      return runError
        ? `Dispersion analysis could not run: ${runError}`
        : "Dispersion analysis could not run.";
    }
    if (runState === "done" && result) {
      return `Dispersion complete for ${result.groups.length} group(s).`;
    }
    return DEFAULT_STATUS;
  }, [runState, runError, result]);

  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 flex flex-col gap-3">
      <h2 className="text-sm font-medium text-white">Dispersion</h2>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Forward Coordinate
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={forward}
            onChange={(e) => setForward(e.target.value)}
          >
            <option value="">(select forward coordinate)</option>
            {coordinateOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Lateral Coordinate
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={lateral}
            onChange={(e) => setLateral(e.target.value)}
          >
            <option value="">(select lateral coordinate)</option>
            {coordinateOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs uppercase tracking-wide text-gray-400">
            Group By
          </span>
          <select
            className="rounded bg-gray-900 border border-gray-700 px-2 py-1"
            value={groupColumn}
            onChange={(e) => setGroupColumn(e.target.value)}
          >
            <option value={ALL_SHOTS}>{ALL_SHOTS}</option>
            {groupOptions.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={handleRun}
          disabled={!canRun || runState === "running"}
          className="self-end rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 text-white px-3 py-2 text-sm font-medium"
        >
          Analyze Dispersion
        </button>
      </div>

      <p
        data-testid="lmd-status"
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
          <div data-testid="lmd-results-table">
            <h3 className="text-xs font-medium text-white mb-1">
              Dispersion by Group
            </h3>
            <ResultsTable groups={result.groups} />
          </div>

          <div className="flex flex-wrap gap-3" data-testid="lmd-ellipse-plots">
            {result.groups.map((group) => (
              <div
                key={group.group}
                className="flex flex-col items-center gap-1"
              >
                <span className="text-xs text-gray-400">{group.group}</span>
                <EllipsePlot result={group} />
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

export default LaunchMonitorDispersionPanel;
