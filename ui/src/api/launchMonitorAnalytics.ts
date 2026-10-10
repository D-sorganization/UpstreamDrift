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
  DispersionPayloadV2,
  FlexibleAnalysisPayload,
  LaunchMonitorAnalysisResultV2,
  MultivariatePayloadV2,
  RelationshipsPayloadV2,
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

/**
 * Group-by candidates accepted by `POST /v2/dispersion`
 * (`DispersionPayloadV2.group_column`'s literal union).
 */
export type DispersionGroupColumn = NonNullable<
  DispersionPayloadV2["group_column"]
>;

/**
 * One group's serialized `DispersionResult` from `_dispersion_result_to_dict`
 * (JSON-safe: NaN/infinite float fields become `null`, never a misleading 0).
 */
export interface DispersionGroupResult {
  group: string;
  sample_count: number;
  center_forward: number | null;
  center_lateral: number | null;
  mean_forward: number | null;
  mean_lateral: number | null;
  ellipse_major: number | null;
  ellipse_minor: number | null;
  ellipse_angle_rad: number | null;
  area_95: number | null;
  radial_rmse: number | null;
  radial_p50: number | null;
  radial_p90: number | null;
}

/** Response body for `POST /v2/dispersion` (`analyze_dispersion_v2`). */
export interface DispersionResponse {
  forward: string;
  lateral: string;
  group_column: DispersionGroupColumn | null;
  groups: DispersionGroupResult[];
}

/**
 * Run the PyQt Dispersion tab's shot-dispersion analysis over caller-supplied
 * inline records, via the same `analyze_dispersion` contract the desktop tab
 * calls (`src/tools/launch_monitor_analytics/gui.py` `_compute_dispersion`).
 */
export async function analyzeDispersionV2(
  records: Record<string, unknown>[],
  forward = "carry_distance",
  lateral = "lateral_carry",
  groupColumn?: DispersionGroupColumn | null,
): Promise<DispersionResponse> {
  const payload: DispersionPayloadV2 = {
    records,
    forward,
    lateral,
    group_column: groupColumn ?? null,
  };
  return apiFetch<DispersionResponse>(`${BASE}/v2/dispersion`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Correlation-method candidates accepted by `RelationshipsPayloadV2.method`
 * (`CorrelationMethod` in `src/tools/launch_monitor_model`); same three
 * choices as the desktop `relationship_method` combo.
 */
export type RelationshipMethod = RelationshipsPayloadV2["method"];

/**
 * A matrix cell is `null` (not `0`) whenever the underlying pair is
 * non-finite — see `_json_safe_float` / `_matrix_to_rows` in
 * `src/api/routes/launch_monitor_analytics.py`.
 */
export type NullableMatrix = (number | null)[][];

/**
 * One screened dependency edge from `CorrelationResult.edges`
 * (`DependencyEdge` in `shared.python.launch_monitor.relationships`).
 */
export interface RelationshipEdge {
  source: string;
  target: string;
  coefficient: number | null;
  p_value: number | null;
  adjusted_p_value: number | null;
  sample_count: number;
  includes_derived_metric: boolean;
  includes_boolean_projection: boolean;
}

/** Response body for `POST /v2/relationships` (`_relationships_result_to_dict`). */
export interface RelationshipsResponse {
  method: string;
  metrics: string[];
  coefficients: NullableMatrix;
  p_values: NullableMatrix;
  adjusted_p_values: NullableMatrix | null;
  pair_counts: number[][];
  partial_coefficients: NullableMatrix | null;
  derived_metrics: string[];
  boolean_projected: string[];
  edges: RelationshipEdge[];
}

/**
 * Run the PyQt Relationships tab's correlation/partial-correlation/dependency
 * -network analysis over caller-supplied inline records, via the same
 * `compute_correlations` contract the desktop tab calls
 * (`src/tools/launch_monitor_analytics/gui.py` `_compute_relationship`).
 * `controls` must already exclude any name also present in `metrics` — the
 * desktop tab drops them before calling `compute_correlations`, and the API
 * drops them again defensively (`RelationshipsPayloadV2.effective_controls`).
 */
export async function analyzeRelationshipsV2(
  records: Record<string, unknown>[],
  metrics: string[],
  method: RelationshipMethod = "pearson",
  controls: string[] = [],
  edgeThreshold = 0.3,
): Promise<RelationshipsResponse> {
  const payload: RelationshipsPayloadV2 = {
    records,
    metrics,
    controls,
    method,
    edge_threshold: edgeThreshold,
  };
  return apiFetch<RelationshipsResponse>(`${BASE}/v2/relationships`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Serialized `PCAResult` from `_pca_result_to_dict` (`POST /v2/multivariate`). */
export interface PCAResultPayload {
  metrics: string[];
  component_names: string[];
  explained_variance_ratio: (number | null)[];
  loadings: NullableMatrix;
  scores: NullableMatrix;
  sample_count: number;
}

/** Serialized `VIFResult` from `_vif_result_to_dict` (`POST /v2/multivariate`). */
export interface VIFResultPayload {
  values: Record<string, number | null>;
  sample_count: number;
  warning_metrics: string[];
}

/** Response body for `POST /v2/multivariate` (`analyze_multivariate_v2`). */
export interface MultivariateResponse {
  pca: PCAResultPayload;
  vif: VIFResultPayload;
}

/**
 * Run the PyQt Relationships tab's PCA/VIF diagnostics over caller-supplied
 * inline records, via the same `compute_pca`/`compute_vif` contract the
 * desktop tab calls (`src/tools/launch_monitor_analytics/gui.py`
 * `_compute_multivariate`).
 */
export async function analyzeMultivariateV2(
  records: Record<string, unknown>[],
  metrics: string[],
): Promise<MultivariateResponse> {
  const payload: MultivariatePayloadV2 = { records, metrics };
  return apiFetch<MultivariateResponse>(`${BASE}/v2/multivariate`, {
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
