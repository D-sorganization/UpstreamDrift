/**
 * Launch Monitor Analytics — Flexible Analysis API client (#11987).
 *
 * Web counterpart of the desktop `FlexibleAnalysisWidget`
 * (`src/tools/launch_monitor_analytics/flexible_analysis_widget.py`). Talks
 * only to the traceable `/v2/analyze` contract — no filesystem or dataset-job
 * paths — so every request is caller-supplied, bounded records.
 */

import { apiFetch } from "./fetch";
import type {
  AnalyzePayloadV2,
  FlexibleAnalysisPayload,
  LaunchMonitorAnalysisResultV2,
} from "./generated/types";

const BASE = "/api/tools/launch-monitor-analytics";

/** Subset of `GET /capabilities` consumed by the Flexible Analysis controls. */
export interface LaunchMonitorAnalyticsCapabilities {
  analysis_modes: FlexibleAnalysisPayload["analysis_mode"][];
  correlation_methods: FlexibleAnalysisPayload["correlation_method"][];
  missing_policies: FlexibleAnalysisPayload["missing_policy"][];
  maximum_inline_records: number;
}

/** One predictor's correlation against the request's outcome. */
export interface FlexibleCorrelationEstimate {
  predictor: string;
  coefficient: number | null;
  p_value: number | null;
  adjusted_p_value: number | null;
  ci_lower: number | null;
  ci_upper: number | null;
  sample_count: number;
  method: string;
  is_boolean_projected: boolean;
}

/** One OLS coefficient (intercept or a named predictor). */
export interface FlexibleCoefficientEstimate {
  estimate: number;
  standard_error: number;
  t_statistic: number;
  p_value: number;
  ci_lower: number;
  ci_upper: number;
}

export interface FlexibleResidualDiagnostics {
  rmse: number;
  mae: number;
  residual_mean: number;
  residual_std: number;
  durbin_watson: number | null;
  jarque_bera_p_value: number;
  influential_count: number;
}

export interface FlexibleRegressionEstimate {
  sample_count: number;
  r_squared: number | null;
  adjusted_r_squared: number | null;
  coefficients: Record<string, FlexibleCoefficientEstimate>;
  residual_diagnostics: FlexibleResidualDiagnostics;
}

export interface FlexibleGroupAnalysis {
  group_value: string;
  row_count: number;
  correlations: FlexibleCorrelationEstimate[];
  regression: FlexibleRegressionEstimate | null;
  warnings: string[];
}

export interface FlexibleAnalysisDatasetSummary {
  row_count: number;
  complete_row_count: number;
  selected_columns: string[];
  monitor_vendors: string[];
  session_ids: string[];
  observation_kinds: string[];
  fingerprint_sha256: string;
}

/**
 * Shape of `LaunchMonitorAnalysisResultV2.analysis` once populated.
 *
 * The OpenAPI contract types this field as an open `Record<string, unknown>`
 * (the Python model is `dict[str, Any]`, see
 * `shared.python.launch_monitor.flexible_analysis.FlexibleAnalysisResult.to_dict`),
 * so this interface documents — but cannot statically guarantee — the actual
 * wire shape. Callers must still treat every field as possibly absent.
 */
export interface FlexibleAnalysisResultPayload {
  dataset: FlexibleAnalysisDatasetSummary;
  correlations: FlexibleCorrelationEstimate[];
  regression: FlexibleRegressionEstimate | null;
  groups: FlexibleGroupAnalysis[];
  units: Record<string, string>;
  warnings: string[];
}

export async function fetchLaunchMonitorAnalyticsCapabilities(): Promise<LaunchMonitorAnalyticsCapabilities> {
  return apiFetch<LaunchMonitorAnalyticsCapabilities>(`${BASE}/capabilities`);
}

/**
 * Run the traceable v2 flexible analysis over caller-supplied inline records.
 *
 * No `context`/`model_provenance` evidence is collected from the browser
 * upload flow, so both are left at their contract defaults.
 */
export async function runFlexibleAnalysisV2(
  records: Record<string, unknown>[],
  analysis: FlexibleAnalysisPayload,
): Promise<LaunchMonitorAnalysisResultV2> {
  const payload: AnalyzePayloadV2 = {
    records,
    analysis,
    model_provenance: [],
  };
  return apiFetch<LaunchMonitorAnalysisResultV2>(`${BASE}/v2/analyze`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Request body for `POST /v2/trend` (`TrendPayloadV2` in
 * `src/api/routes/launch_monitor_analytics.py`). Mirrors the PyQt Trends tab
 * (`_TrendParams` / `_read_trend_params` in `gui.py`): `time_column` and
 * `rolling_window` share that widget's defaults, `"captured_at"` and `10`,
 * and the same `[3, 500]` rolling-window bound.
 */
export interface TrendRequest {
  records: Record<string, unknown>[];
  metric: string;
  time_column?: string;
  rolling_window?: number;
}

/** One rolling-statistics row from `TrendResponse.rolling` (JSON-safe: never a misleading 0). */
export interface TrendRollingPoint {
  value: number | null;
  rolling_mean: number | null;
  rolling_median: number | null;
  rolling_std: number | null;
  ewma: number | null;
  /** The request's `time_column`, serialized as an ISO-8601 timestamp string. */
  [timeColumn: string]: number | string | null;
}

/** One ranked step-change candidate from `TrendResponse.change_candidates`. */
export interface TrendChangeCandidate {
  captured_at: string;
  row_index: number;
  before_mean: number | null;
  after_mean: number | null;
  effect_size: number | null;
}

/** Response body for `POST /v2/trend` (`_trend_result_to_dict`). */
export interface TrendResponse {
  metric: string;
  sample_count: number;
  slope_per_day: number | null;
  robust_slope_per_day: number | null;
  p_value: number | null;
  earliest_mean: number | null;
  latest_mean: number | null;
  rolling: TrendRollingPoint[];
  change_candidates: TrendChangeCandidate[];
}

/**
 * Run the PyQt Trends tab's longitudinal trend analysis over caller-supplied
 * inline records, via the same `analyze_trend` contract the desktop tab calls.
 */
export async function postTrend(
  records: Record<string, unknown>[],
  metric: string,
  timeColumn = "captured_at",
  rollingWindow = 10,
): Promise<TrendResponse> {
  const payload: TrendRequest = {
    records,
    metric,
    time_column: timeColumn,
    rolling_window: rollingWindow,
  };
  return apiFetch<TrendResponse>(`${BASE}/v2/trend`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Render a nullable statistic as the API would — never a misleading zero. */
export function formatStat(
  value: number | null | undefined,
  digits = 4,
): string {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(digits);
}
