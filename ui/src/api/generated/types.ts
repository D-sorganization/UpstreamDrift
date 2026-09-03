/**
 * AUTO-GENERATED FILE - DO NOT EDIT BY HAND.
 *
 * TypeScript mirror of the API contract defined by the Pydantic models in
 * src/api/models/ and the FastAPI route response models (src/api/server.py).
 *
 * Regenerate with:
 *     python scripts/generate_ui_api_types.py
 *
 * Freshness is enforced by tests/api/test_generated_ui_api_types.py.
 * See issue #7447.
 */

/**
 * A single AIP server capability. See issue #763
 */
export interface AIPCapability {
  /** Capability name */
  name: string;
  /** Capability version */
  version: string;
  /** Supported methods */
  methods: string[];
}

/**
 * Response model for AIP capability negotiation. See issue #763
 */
export interface AIPHandshakeResponse {
  /** Server identifier */
  server_name: string;
  /** JSON-RPC version */
  protocol_version: string;
  /** Available capabilities */
  capabilities: AIPCapability[];
  /** All supported RPC method names */
  supported_methods: string[];
}

/**
 * API key creation model.
 */
export interface APIKeyCreate {
  /** Friendly name for the API key */
  name: string;
}

/**
 * API key response model.
 */
export interface APIKeyResponse {
  id: number;
  name: string;
  key?: string | null;
  is_active: boolean;
  last_used?: string | null;
  usage_count: number;
  created_at: string;
  expires_at?: string | null;
}

/**
 * Response for the active theme.
 */
export interface ActiveThemeResponse {
  name: string;
  is_builtin: boolean;
  colors: Record<string, string>;
}

/**
 * Request model for sending commands to multiple actuators at once. See issue #1198
 */
export interface ActuatorBatchCommandRequest {
  /** List of actuator commands */
  commands: ActuatorCommandRequest[];
}

/**
 * Request model for sending commands to individual actuators. Preconditions: - actuator_index must be non-negative - control_type must be a known type See issue #1198
 */
export interface ActuatorCommandRequest {
  /** Index of the actuator to command */
  actuator_index: number;
  /** Command value (meaning depends on control_type) */
  value: number;
  /** Control type: constant, polynomial, pd_gains, trajectory */
  control_type: string;
  /** Additional parameters (e.g., polynomial coefficients, PD gains) */
  parameters?: Record<string, unknown> | null;
}

/**
 * Response model for actuator command acknowledgment. See issue #1198
 */
export interface ActuatorCommandResponse {
  /** Actuator that was commanded */
  actuator_index: number;
  /** Value actually applied */
  applied_value: number;
  /** Control type used */
  control_type: string;
  /** Status message */
  status: string;
  /** Whether value was clamped to limits */
  clamped: boolean;
}

/**
 * Descriptor for a single actuator. See issue #1198
 */
export interface ActuatorInfo {
  /** Actuator index */
  index: number;
  /** Actuator/joint name */
  name: string;
  /** Active control type */
  control_type: string;
  /** Current command value */
  value: number;
  /** Minimum allowed value */
  min_value: number;
  /** Maximum allowed value */
  max_value: number;
  /** Physical units */
  units: string;
  /** Associated joint type */
  joint_type: string;
}

/**
 * Response model for the actuator control panel state. See issue #1198
 */
export interface ActuatorPanelResponse {
  /** Total actuator count */
  n_actuators: number;
  /** Per-actuator descriptors */
  actuators: ActuatorInfo[];
  /** Supported control types */
  available_control_types?: string[];
  /** Active engine name */
  engine_name: string;
}

/**
 * Response model for actuator/control state. See issue #1209
 */
export interface ActuatorStateResponse {
  /** Active control strategy */
  strategy: string;
  /** Number of actuated joints */
  n_joints: number;
  /** Names of all joints */
  joint_names: string[];
  /** Current applied torques */
  torques: number[];
  /** Target positions if set */
  target_positions?: number[] | null;
  /** Target velocities if set */
  target_velocities?: number[] | null;
  /** Proportional gains */
  kp: number[];
  /** Derivative gains */
  kd: number[];
  /** Integral gains */
  ki: number[];
  /** Per-joint details */
  joints: JointInfoResponse[];
  /** All available control strategies */
  available_strategies: Record<string, string>[];
}

/**
 * Request model for per-actuator parameter updates. Preconditions: - strategy must be a known control strategy - torques, if provided, must be a list of floats - gains must be positive when provided See issue #1209
 */
export interface ActuatorUpdateRequest {
  /** Control strategy (pd, pid, zero, etc.) */
  strategy?: string | null;
  /** Per-joint torque values */
  torques?: number[] | null;
  /** Proportional gain(s) */
  kp?: number | number[] | null;
  /** Derivative gain(s) */
  kd?: number | number[] | null;
  /** Integral gain(s) */
  ki?: number | number[] | null;
  /** Target joint positions (rad) */
  target_positions?: number[] | null;
  /** Target joint velocities (rad/s) */
  target_velocities?: number[] | null;
}

/**
 * Caller-supplied lineage and identity assertions.
 */
export interface AnalysisContextV2 {
  authority?: DatasetAuthorityV2 | null;
  player_identity?: PlayerIdentityV2;
  session_identity?: SessionIdentityV2;
  order_evidence?: OrderEvidenceV2;
  transformations: TransformRecordV2[];
  sources: SourceFileReferenceV2[];
  source_units?: Record<string, string>;
}

export interface AnalysisLineageV2 {
  dataset_fingerprint_sha256: string;
  authority?: DatasetAuthorityV2 | null;
  transformations: TransformRecordV2[];
  sources: SourceFileReferenceV2[];
  backing_records: BackingRecordV2[];
}

/**
 * Response model for aggregated analysis metrics. Provides statistical summary of simulation metrics over time. See issue #1203
 */
export interface AnalysisMetricsSummary {
  /** Name of the metric */
  metric_name: string;
  /** Current value */
  current: number;
  /** Minimum observed value */
  minimum: number;
  /** Maximum observed value */
  maximum: number;
  /** Mean value over time window */
  mean: number;
  /** Standard deviation */
  std_dev: number;
}

/**
 * Request model for biomechanical analysis. Preconditions: - analysis_type must be a known analysis type - export_format must be a supported format
 */
export interface AnalysisRequest {
  /** Type of analysis (kinematics, kinetics, etc.) */
  analysis_type: string;
  /** Source of data (simulation, c3d, video) */
  data_source: string;
  /** Analysis parameters */
  parameters?: Record<string, unknown>;
  /** Optional analysis input data such as trajectories or timebases */
  data?: Record<string, unknown>;
  /** Output format */
  export_format: string;
}

/**
 * Response model for biomechanical analysis. Postconditions: - analysis_type must match the original request - results must be non-empty on success
 */
export interface AnalysisResponse {
  /** Type of analysis performed */
  analysis_type: string;
  /** Whether analysis completed successfully */
  success: boolean;
  /** Analysis results */
  results: Record<string, unknown>;
  /** Generated visualization files */
  visualizations?: string[] | null;
  /** Path to exported results */
  export_path?: string | null;
}

/**
 * Response model for analysis statistics endpoint. See issue #1203
 */
export interface AnalysisStatisticsResponse {
  /** Current simulation time */
  sim_time: number;
  /** Number of samples collected */
  sample_count: number;
  /** Statistical summaries per metric */
  metrics: AnalysisMetricsSummary[];
  /** Time series data for plotting (metric_name -> values) */
  time_series?: Record<string, number[]> | null;
}

/**
 * Bounded inline data and an analysis request; paths are never accepted.
 */
export interface AnalyzePayload {
  records: Record<string, unknown>[];
  analysis: FlexibleAnalysisPayload;
}

/**
 * V2 request with explicit dataset, transformation, and identity context.
 */
export interface AnalyzePayloadV2 {
  records: Record<string, unknown>[];
  analysis: FlexibleAnalysisPayload;
  context?: AnalysisContextV2;
  model_provenance: ModelProvenanceV2[];
}

/**
 * Appearance preferences for the web shell.
 */
export interface AppearanceSettings {
  /** Preferred theme name (must match a theme from GET /themes). */
  theme_id: string;
  /** Root font scale multiplier (0.5–2.0). */
  font_scale: number;
}

/**
 * One descriptive association or a typed unavailable state.
 */
export interface AssociationEstimateV1 {
  state: "available" | "unavailable";
  reason_code?: "insufficient_samples" | "insufficient_groups" | "constant_x" | "constant_y" | "constant_both" | null;
  sample_count: number;
  group_count: number;
  pearson_r?: number | null;
  spearman_r?: number | null;
  slope?: number | null;
  intercept?: number | null;
  r_squared?: number | null;
  ci_lower?: number | null;
  ci_upper?: number | null;
  interval_withheld_reason?: "insufficient_degrees_of_freedom" | "clustered_observations" | null;
}

export interface AvailabilityV1 {
  state: "available" | "unavailable";
  reason_code?: string | null;
  message?: string | null;
  observed_count: number;
  required_count: number;
}

export interface AvailabilityV2 {
  result_path: string;
  state: "available" | "unavailable";
  reason_code?: string | null;
  message?: string | null;
  observed_count?: number | null;
  required_count?: number | null;
}

/**
 * Stable reference to exactly one input record without copying values.
 */
export interface BackingRecordV2 {
  record_sha256: string;
  shot_id?: string | null;
  session_id?: string | null;
  source_row?: number | string | null;
  source_id?: string | null;
  unlinked_reason?: "no_source_reference_declared" | "session_not_linked_to_source_reference" | null;
}

/**
 * Trajectory and summary metrics for a single flight model.
 */
export interface BallFlightModelResult {
  model_name: string;
  model_key: FlightModelType;
  coefficients: Record<string, number>;
  trajectory: BallFlightTrajectorySample[];
  summary: BallFlightSummary;
}

/**
 * Request to simulate a single ball-flight trajectory.
 */
export interface BallFlightSimulationRequest {
  /** Initial ball speed [m/s] */
  ball_speed_mps: number;
  /** Vertical launch angle [deg] */
  launch_angle_deg: number;
  /** Horizontal launch azimuth [deg] */
  azimuth_angle_deg: number;
  /** Initial spin rate [rpm] */
  spin_rate_rpm: number;
  /** Spin-axis tilt [deg] */
  spin_axis_tilt_deg: number;
  /** Wind speed [m/s] */
  wind_speed_mps: number;
  /** Wind direction [deg] */
  wind_direction_deg: number;
  /** Ball-flight model identifier */
  model_name: FlightModelType;
  /** Optional list of flight models for overlay comparison. When provided, the response carries one result per (deduplicated) model; the top-level trajectory/summary mirror the first entry for backwards compatibility. */
  models?: FlightModelType[] | null;
  /** Maximum simulation time [s] */
  max_time_s: number;
  /** Returned trajectory sample interval [s] */
  time_step_s: number;
}

/**
 * Response containing ball-flight trajectories and summary metrics. The top-level ``model_name``/``trajectory``/``summary`` fields describe the first requested model (backwards compatible with single-model clients); ``results`` carries one entry per requested model for overlay comparison (issue #7456).
 */
export interface BallFlightSimulationResponse {
  model_name: string;
  model_key: FlightModelType;
  coefficients: Record<string, number>;
  trajectory: BallFlightTrajectorySample[];
  summary: BallFlightSummary;
  results: BallFlightModelResult[];
}

/**
 * Scalar trajectory metrics.
 */
export interface BallFlightSummary {
  carry_m: number;
  apex_m: number;
  flight_time_s: number;
  landing_angle_deg: number;
  lateral_deviation_m: number;
}

/**
 * Single sampled trajectory point.
 */
export interface BallFlightTrajectorySample {
  time_s: number;
  position_m: number[];
  velocity_mps: number[];
}

export interface BaselineProvenanceV1 {
  baseline_id: string;
  version: string;
  source_url: string;
  license: string;
  table_sha256: string;
  contract_version: string;
}

/**
 * Response model for biomechanics metrics. See issue #1209
 */
export interface BiomechanicsMetricsResponse {
  /** Current simulation time */
  sim_time: number;
  /** Club head speed (m/s) */
  club_head_speed?: number | null;
  /** Total kinetic energy (J) */
  kinetic_energy?: number | null;
  /** Total potential energy (J) */
  potential_energy?: number | null;
  /** Current joint positions */
  joint_positions: number[];
  /** Current joint velocities */
  joint_velocities: number[];
  /** Peak torque across all joints (N*m) */
  peak_torque?: number | null;
  /** Sum of absolute torques (N*m) */
  total_torque_magnitude?: number | null;
}

/**
 * Response model for body positioning. See issue #1179
 */
export interface BodyPositionResponse {
  /** Name of the positioned body */
  body_name: string;
  /** Applied position [x, y, z] */
  position: number[];
  /** Applied rotation [roll, pitch, yaw] */
  rotation: number[];
  /** Status message */
  status: string;
}

/**
 * Request model for updating body position in simulation. See issue #1179
 */
export interface BodyPositionUpdateRequest {
  /** Name of the body to reposition */
  body_name: string;
  /** New position [x, y, z] */
  position?: number[] | null;
  /** New rotation [roll, pitch, yaw] in radians */
  rotation?: number[] | null;
}

export interface Body_analyze_video_analyze_video_post {
  file: string;
}

export interface Body_analyze_video_api_analyze_video_post {
  file: string;
}

export interface Body_analyze_video_api_v1_analyze_video_post {
  file: string;
}

export interface Body_analyze_video_async_analyze_video_async_post {
  file: string;
}

export interface Body_analyze_video_async_api_analyze_video_async_post {
  file: string;
}

export interface Body_analyze_video_async_api_v1_analyze_video_async_post {
  file: string;
}

export interface Body_import_dataset_api_tools_data_explorer_import_post {
  file: string;
}

export interface Body_import_dataset_api_v1_tools_data_explorer_import_post {
  file: string;
}

export interface Body_import_dataset_tools_data_explorer_import_post {
  file: string;
}

export interface Body_upload_c3d_api_tools_motion_capture_upload_c3d_post {
  file: string;
}

export interface Body_upload_c3d_api_v1_tools_motion_capture_upload_c3d_post {
  file: string;
}

export interface Body_upload_c3d_tools_motion_capture_upload_c3d_post {
  file: string;
}

/**
 * Metadata extracted from an uploaded C3D file. Marker positions are converted to meters server-side (mirroring the desktop C3D viewer's ``target_units="m"`` handling) so the web visualizer never has to guess mm-vs-m scaling.
 */
export interface C3DUploadResponse {
  recording_name: string;
  marker_names: string[];
  frame_rate: number;
  total_frames: number;
  duration_seconds: number;
  /** POINT units declared in the file ('' when absent) */
  native_units: string;
  /** Units of the stored marker positions */
  converted_units: string;
}

/**
 * Response model for enumerating available camera presets. See issue #7452
 */
export interface CameraPresetListResponse {
  /** Available camera presets with their view vectors */
  presets: CameraPresetResponse[];
}

/**
 * Request model for camera preset selection. Preconditions: - preset must be a known camera preset See issue #1202
 */
export interface CameraPresetRequest {
  /** Camera preset (side, front, top, follow_ball, follow_club) */
  preset: string;
}

/**
 * Response model for camera preset application. See issue #1202
 */
export interface CameraPresetResponse {
  /** Applied camera preset */
  preset: string;
  /** Camera position [x, y, z] */
  position: number[];
  /** Camera look-at target [x, y, z] */
  target: number[];
  /** Camera up vector [x, y, z] */
  up: number[];
}

/**
 * Availability report for one canonical-core workspace.
 */
export interface CanonicalCoreStatus {
  /** Registry id, e.g. canonical_core_estimation */
  tool_id: string;
  /** Workspace mode: estimation or comparison */
  mode: string;
  /** Human-readable workspace name */
  name: string;
  /** Workspace description from the registry */
  description: string;
  /** React route that renders this workspace */
  web_route: string;
  /** Capability tags declared by the registry descriptor */
  capabilities?: string[];
  /** True once a compute service backs this workspace */
  available: boolean;
  /** Why the workspace is unavailable; empty when available */
  reason: string;
  /** Actionable guidance for the user; empty when available */
  next_step: string;
}

/**
 * Availability report for every canonical-core workspace.
 */
export interface CanonicalCoreStatusList {
  workspaces?: CanonicalCoreStatus[];
}

/**
 * Response model for a single capability level.
 */
export interface CapabilityLevelResponse {
  /** Capability name */
  name: string;
  /** Support level: full, partial, or none */
  level: "full" | "partial" | "none";
  /** Whether capability is available */
  supported: boolean;
}

/**
 * Counts of capabilities per support level (issue #7447). Typed (rather than ``dict[str, int]``) so the generated TypeScript contract exposes the exact keys the UI reads.
 */
export interface CapabilitySummaryResponse {
  /** Number of fully supported capabilities */
  full: number;
  /** Number of partially supported capabilities */
  partial: number;
  /** Number of unsupported capabilities */
  none: number;
}

/**
 * Request to start a capture session.
 */
export interface CaptureSessionRequest {
  /** Capture source: c3d, openpose, mediapipe */
  source_type: string;
  /** Target frame rate */
  frame_rate: number;
}

/**
 * Response after starting/stopping a capture session.
 */
export interface CaptureSessionResponse {
  session_id: string;
  status: string;
  source_type: string;
  message: string;
}

/**
 * Available motion capture source.
 */
export interface CaptureSource {
  id: string;
  name: string;
  /** c3d, openpose, or mediapipe */
  type: string;
  available: boolean;
  /** Why the source is unavailable (None when available) */
  reason?: string | null;
  description: string;
}

/**
 * Request model for Character Builder URDF generation. Preconditions: - height_m must be in [1.5, 2.1] - mass_kg must be in [40, 150] - build_type must be athletic, average, heavy, or slim
 */
export interface CharacterBuilderRequest {
  /** Height in meters */
  height_m: number;
  /** Weight in kilograms */
  mass_kg: number;
  /** Build type */
  build_type: string;
}

export interface ClaimsV2 {
  vendor_comparison: "descriptive" | "matched_agreement";
  device_emulation: boolean;
  device_certification: boolean;
  causal_inference: boolean;
}

export interface ConfidenceIntervalV1 {
  lower: number;
  upper: number;
  level: number;
  method: string;
}

/**
 * Request body for deterministic contraction-rate estimation.
 */
export interface ContractionEstimateRequest {
  decay_rate: number;
  dimension: number;
  horizon: number;
  n_steps: number;
  n_trials: number;
  perturbation_scale: number;
}

/**
 * Response model for control features registry data. See issue #1209
 */
export interface ControlFeaturesResponse {
  /** Engine class name */
  engine: string;
  /** Total registered features */
  total_features: number;
  /** Available on this engine */
  available_features: number;
  /** Feature categories with counts */
  categories: Record<string, unknown>[];
  /** Feature descriptors */
  features: Record<string, unknown>[];
}

/**
 * Request model for setting control state.
 */
export interface ControlStateRequest {
  /** Control strategy */
  strategy: string;
  /** Direct torque values */
  torques?: number[] | null;
  /** Joint index for single-joint control */
  joint_index?: number | null;
  /** Torque for single joint */
  joint_torque?: number | null;
  /** Proportional gain */
  kp?: number | null;
  /** Derivative gain */
  kd?: number | null;
  /** Integral gain */
  ki?: number | null;
  /** Target positions */
  target_positions?: number[] | null;
  /** Target velocities */
  target_velocities?: number[] | null;
}

/**
 * Request model for counterfactual / induced-acceleration analysis. Preconditions: - kind must be a known counterfactual kind (see ``src.shared.python.analysis.orchestrator`` — single source). See issue #7450.
 */
export interface CounterfactualRequest {
  /** Counterfactual kind: 'ztcf' / 'zvcf' (counterfactual accelerations) or 'gravity' / 'drift' / 'control' / 'total' (induced accelerations) */
  kind: string;
  /** When true and no counterfactual data is stored yet, replay the recorded frames through the engine (expensive) */
  run_post_hoc: boolean;
}

export interface CourseStateColumnsV1 {
  lie_column: string;
  context_column: string;
  target_column: string;
  distance_column: string;
  distance_unit: "yd" | "m";
}

export interface CourseStateValueV1 {
  lie: string;
  context: string;
  target: string;
  distance_yards: number;
}

/**
 * Row and player exclusions used by a selected-pair analysis.
 */
export interface CovariationMissingnessV1 {
  input_row_count: number;
  pairwise_complete_row_count: number;
  missing_by_variable: Record<string, number>;
  non_numeric_by_variable: Record<string, number>;
  non_finite_by_variable: Record<string, number>;
  excluded_by_reason: Record<string, number>;
  eligible_player_count: number;
  excluded_player_count_by_reason: Record<string, number>;
  policy: "pairwise";
}

/**
 * One deterministically ranked pair from an exploratory scan.
 */
export interface CovariationPairRankV1 {
  rank: number;
  state: "available" | "unavailable";
  reason_code?: "insufficient_eligible_players" | null;
  x_column: string;
  y_column: string;
  x_unit: MetricUnitsV2;
  y_unit: MetricUnitsV2;
  random_effect_r?: number | null;
  fixed_effect_r?: number | null;
  within_player_r?: number | null;
  between_player_r?: number | null;
  contributor_count: number;
  total_sample_count: number;
  input_row_count: number;
  pairwise_complete_row_count: number;
  excluded_row_count: number;
  i_squared_pct?: number | null;
  direction_consistency?: number | null;
}

/**
 * Named uncertainty methods and their scientific limits.
 */
export interface CovariationUncertaintyV1 {
  confidence_level: number;
  per_player_interval: "fisher-z";
  pooled_interval: "fisher-z-unclustered";
  within_player_interval: "unavailable-clustered";
  between_player_interval: "fisher-z-above-min-groups";
  between_player_interval_min_groups: number;
  fixed_effect_method: "inverse-variance-fisher-z";
  random_effect_method: "dersimonian-laird-fisher-z";
  assumptions: string[];
}

/**
 * Request to create a terrain environment.
 */
export interface CreateEnvironmentRequest {
  /** Preset name (putting_green, fairway, driving_range, etc.) */
  preset: string;
  /** Override width (meters) */
  width?: number | null;
  /** Override length (meters) */
  length?: number | null;
  /** Slope angle (degrees) */
  slope_angle_deg: number;
  /** Slope direction (degrees) */
  slope_direction_deg: number;
}

/**
 * Perturbation study configuration. All fields match ``CrossEngineSimConfig`` from the service layer.
 */
export interface CrossEnginePerturbationConfig {
  /** Simulation horizon (seconds) */
  t_end: number;
  /** Integration timestep (seconds) */
  dt: number;
  /** Perturbation amplitude */
  noise_amplitude: number;
  /** Number of perturbation trials */
  n_trials: number;
  /** Random seed for reproducibility */
  seed: number;
}

/**
 * Request body for POST /analysis/cross-engine.
 */
export interface CrossEngineStudyRequest {
  /** Engine names to compare; each must be a recognised engine. */
  engines?: string[];
  config: CrossEnginePerturbationConfig;
  /** When false, the study fails instead of silently running a 2-DOF stub for a requested engine whose real backend is unavailable (#8817). Either way the result declares each engine's backend ('real' or 'stub_2dof') and lists 'stubbed_engines'. */
  allow_stub_substitution: boolean;
}

/**
 * Request model for data export. Preconditions: - format must be csv or json See issue #1203
 */
export interface DataExportRequest {
  /** Export format (csv, json) */
  format: string;
  /** Include metrics data */
  include_metrics: boolean;
  /** Include time series */
  include_time_series: boolean;
  /** Optional time range [start, end] in seconds */
  time_range?: number[] | null;
}

/**
 * Commit-addressable authority for the analyzed dataset.
 */
export interface DatasetAuthorityV2 {
  dataset_id: string;
  repository?: string | null;
  commit?: string | null;
  dataset_path?: string | null;
  manifest_sha256?: string | null;
}

/**
 * A single dataset-generation parameter, described for the UI. Mirrors the ``DatasetControl`` interface in ``ui/src/api/useDatasetGenerator.ts``.
 */
export interface DatasetControl {
  /** Field name on DatasetGenerationRequest */
  id: string;
  /** Human-readable label */
  name: string;
  /** Widget type: select, range, or text */
  type: string;
  /** Default value */
  value: unknown;
  /** Choices for select */
  options?: string[] | null;
  /** Minimum for range widgets */
  min?: number | null;
  /** Maximum for range widgets */
  max?: number | null;
  /** Step for range widgets */
  step?: number | null;
}

/**
 * Response for the dataset generation control catalog.
 */
export interface DatasetControlListResponse {
  controls: DatasetControl[];
}

/**
 * Request to filter dataset rows.
 */
export interface DatasetFilterRequest {
  /** Column name to filter on */
  column: string;
  /** Filter operator: eq, ne, gt, lt, gte, lte, contains */
  operator: "eq" | "ne" | "gt" | "lt" | "gte" | "lte" | "contains";
  /** Filter value (string-encoded) */
  value: string;
  limit: number;
}

/**
 * Request model for dataset generation.
 */
export interface DatasetGenerationRequest {
  /** Number of simulation runs */
  num_samples: number;
  /** Duration per simulation (seconds) */
  duration: number;
  /** Simulation timestep */
  timestep: number;
  /** Random seed for reproducibility */
  seed: number;
  /** Randomize initial positions */
  vary_positions: boolean;
  /** Randomize initial velocities */
  vary_velocities: boolean;
  /** Record inertia matrices */
  record_mass_matrix: boolean;
  /** Record bias/gravity forces */
  record_dynamics: boolean;
  /** Record drift/control decomposition */
  record_drift_control: boolean;
  /** Export format (hdf5, sqlite, csv) */
  export_format: string;
  /** Output path */
  output_path: string;
}

/**
 * Response model for dataset generation.
 */
export interface DatasetGenerationResponse {
  status: string;
  num_samples: number;
  total_frames: number;
  export_path: string;
  export_format: string;
}

/**
 * Information about a discovered dataset file.
 */
export interface DatasetInfo {
  dataset_id?: string | null;
  name: string;
  path: string;
  format: string;
  size_bytes: number;
  columns?: string[];
}

/**
 * A private-data job request containing no inline observations.
 */
export interface DatasetJobRequestV1 {
  contract_version: "launch-monitor-dataset-job/1.0.0";
  dataset: DatasetReferenceV1;
  operation: DatasetOperationV1;
}

/**
 * Bounded page of aggregate or source-backing records, never shot rows.
 */
export interface DatasetJobResultPageV1 {
  contract_version: "launch-monitor-dataset-job/1.0.0";
  job_id: string;
  offset: number;
  limit: number;
  total_items: number;
  next_offset?: number | null;
  items: Record<string, unknown>[];
}

/**
 * Data-free state and aggregate counts for one server-side job.
 */
export interface DatasetJobStatusV1 {
  contract_version: "launch-monitor-dataset-job/1.0.0";
  job_id: string;
  status: "queued" | "running" | "completed" | "unavailable" | "failed";
  submitted_at_utc: string;
  completed_at_utc?: string | null;
  input_row_count: number;
  result_item_count: number;
  unavailable?: DatasetUnavailableStateV1 | null;
}

/**
 * Response listing available datasets. ``total`` reports the number of entries returned in this (paginated) page, preserving the historical field semantics. ``offset``/``limit`` echo the pagination window and ``truncated`` flags when the on-disk scan hit the hard cap and more files may exist beyond the returned page (#7740 H).
 */
export interface DatasetListResponse {
  datasets: DatasetInfo[];
  total: number;
  search_dir: string;
  offset: number;
  limit: number;
  truncated: boolean;
}

/**
 * Allow-listed aggregate operation; arbitrary query text is forbidden.
 */
export interface DatasetOperationV1 {
  kind: "source_summary" | "metric_summary" | "correlation";
  metrics: string[];
  group_by?: "source_id" | "monitor" | "club" | null;
  minimum_group_rows: number;
}

/**
 * Response with a preview of dataset contents.
 */
export interface DatasetPreviewResponse {
  name: string;
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number;
  format: string;
}

/**
 * Content-addressed reference to one authorized authority checkout.
 */
export interface DatasetReferenceV1 {
  root_id: string;
  repository: string;
  commit: string;
  manifest_sha256: string;
  content_sha256: string;
  expected_row_count: number;
}

/**
 * Paginated rows from durable dataset storage (issue #6991).
 */
export interface DatasetRowsResponse {
  dataset_id: string;
  columns: string[];
  rows: Record<string, unknown>[];
  offset: number;
  limit: number;
  total_rows: number;
}

/**
 * Response with summary statistics for a dataset.
 */
export interface DatasetStatsResponse {
  name: string;
  columns: string[];
  row_count: number;
  stats: Record<string, Record<string, number | null>>;
}

/**
 * Data-free reason that an immutable reference cannot be used.
 */
export interface DatasetUnavailableStateV1 {
  code: "root_not_authorized" | "authority_unavailable" | "repository_mismatch" | "commit_mismatch" | "manifest_mismatch" | "content_mismatch" | "row_count_mismatch" | "backing_manifest_mismatch" | "dependency_unavailable" | "operation_unavailable" | "internal_execution_error";
  message: string;
  retryable: boolean;
}

/**
 * Request body for realized drift-to-input ratio analysis.
 */
export interface DriftControlRatioRequest {
  drift_generalized_force: number[][];
  control_generalized_force: number[][];
  epsilon: number;
}

/**
 * Response model for engine capabilities. See issue #1204
 */
export interface EngineCapabilitiesResponse {
  /** Engine identifier */
  engine_name: string;
  /** Engine type enum value */
  engine_type: string;
  /** All capabilities with support levels */
  capabilities: CapabilityLevelResponse[];
  /** Counts: full, partial, none */
  summary: CapabilitySummaryResponse;
}

export interface EngineListResponse {
  engines: EngineStatusResponse[];
  mode: string;
}

/**
 * Response for ``POST /engines/{engine_name}/load`` (lazy-loading UI).
 */
export interface EngineLoadResponse {
  /** Load status, e.g. 'loaded' */
  status: string;
  /** Engine name that was loaded */
  engine: string;
  /** Engine version if known */
  version?: string | null;
  /** Engine capability tags */
  capabilities?: string[];
  /** Human-readable status message */
  message?: string | null;
}

/**
 * Response for ``GET /engines/{engine_name}/probe`` (lazy-loading UI).
 */
export interface EngineProbeResponse {
  /** Whether the engine is available */
  available: boolean;
  /** Engine version if available */
  version?: string | null;
  /** Engine capability tags */
  capabilities?: string[];
  /** Probe failure detail */
  error?: string | null;
}

/**
 * Response model for engine status.
 */
export interface EngineStatusResponse {
  /** Engine name identifier */
  name: string;
  /** Whether engine is available */
  available: boolean;
  /** Whether engine is currently loaded */
  loaded: boolean;
  /** Engine version if available */
  version?: string | null;
  /** Engine capabilities */
  capabilities?: string[];
  /** Engine type identifier (deprecated) */
  engine_type: string;
  /** Current status */
  status: string;
  /** Whether engine is available (deprecated) */
  is_available: boolean;
  /** Engine description */
  description: string;
}

/**
 * An available environment preset.
 */
export interface EnvironmentPreset {
  /** Preset identifier */
  name: string;
  /** Human-readable description */
  description: string;
  /** Terrain types in this environment */
  terrain_types: string[];
  /** Width in meters */
  width_m: number;
  /** Length in meters */
  length_m: number;
}

export interface EstimateSummaryV1 {
  count: number;
  mean?: number | null;
  standard_deviation?: number | null;
  standard_error?: number | null;
  confidence_interval?: ConfidenceIntervalV1 | null;
}

export interface ExcludedRowV1 {
  source_index: number;
  shot_id?: string | null;
  reason_code: "missing_course_state" | "invalid_distance" | "outside_baseline";
  message: string;
}

export interface ExclusionSummaryV1 {
  input_row_count: number;
  included_row_count: number;
  total_excluded: number;
  by_reason: Record<string, number>;
}

/**
 * Hash-verified expected-strokes benchmark and publication metadata.
 */
export interface ExpectedStrokesBaselineV2 {
  contract_version: "launch-monitor-strokes-gained-baseline/2.0.0";
  baseline_id: string;
  version: string;
  source_url: string;
  license: string;
  table_sha256: string;
  states: ExpectedStrokesStateV2[];
}

/**
 * One benchmark point for an explicit target-aware course state.
 */
export interface ExpectedStrokesStateV2 {
  lie: string;
  context: string;
  target: string;
  distance_yards: number;
  expected_strokes: number;
  standard_error?: number | null;
}

/**
 * Request model for executing a control feature.
 */
export interface FeatureExecuteRequest {
  /** Feature method name */
  feature_name: string;
  /** Feature arguments */
  args?: Record<string, unknown>;
}

/**
 * Pydantic mirror of :class:`FeatureReport` for the OpenAPI schema.
 */
export interface FeatureReportModel {
  name: string;
  display_name: string;
  available: boolean;
  version?: string | null;
  tier: string;
  docker_stage?: string | null;
  install_channel: string;
  install_command: string;
  pip_extra?: string | null;
  approx_size_mb: number;
  message: string;
  missing?: string[];
  depends_on?: string[];
}

/**
 * Serialized form of :class:`FlexibleAnalysisRequest`.
 */
export interface FlexibleAnalysisPayload {
  outcome: string;
  predictors: string[];
  analysis_mode: "correlation" | "regression" | "comprehensive";
  correlation_method: "pearson" | "spearman" | "kendall";
  missing_policy: "pairwise" | "listwise" | "fail";
  group_by?: string | null;
  confidence_level: number;
  min_samples: number;
  allow_aggregate: boolean;
}

/**
 * Metadata describing one registered ball-flight model.
 */
export interface FlightModelInfo {
  key: FlightModelType;
  name: string;
  description: string;
  reference: string;
}

/**
 * Enumeration of every flight model in :class:`FlightModelRegistry`.
 */
export interface FlightModelListResponse {
  models: FlightModelInfo[];
}

/**
 * Available ball flight physics models.
 */
export type FlightModelType = "waterloo_penner" | "macdonald_hanzely" | "nathan" | "ballantyne" | "jcole" | "rospie_dl" | "charry_l3";

/**
 * Request model for enabling/configuring force/torque overlays. Preconditions: - force_types must each be a known force type - scale_factor must be positive See issue #1199
 */
export interface ForceOverlayRequest {
  /** Whether overlays are visible */
  enabled: boolean;
  /** Which force types to display */
  force_types?: string[];
  /** Arrow length scaling factor */
  scale_factor: number;
  /** Color-code arrows by magnitude */
  color_by_magnitude: boolean;
  /** Only show forces on these bodies (None = all) */
  body_filter?: string[] | null;
  /** Show magnitude labels */
  show_labels: boolean;
}

/**
 * Response model for force/torque overlay data. See issue #1199
 */
export interface ForceOverlayResponse {
  /** Current simulation time */
  sim_time: number;
  /** All active force vectors */
  vectors?: ForceVector3D[];
  /** Sum of all force magnitudes */
  total_force_magnitude: number;
  /** Sum of all torque magnitudes */
  total_torque_magnitude: number;
  /** Current overlay configuration */
  overlay_config?: Record<string, unknown>;
}

/**
 * A single force/torque vector for 3D overlay rendering. See issue #1199
 */
export interface ForceVector3D {
  /** Body this force acts on */
  body_name: string;
  /** Type: applied, gravity, contact, bias */
  force_type: string;
  /** Application point [x, y, z] */
  origin: number[];
  /** Force direction [dx, dy, dz] */
  direction: number[];
  /** Force magnitude (N or N*m) */
  magnitude: number;
  /** RGBA color for rendering */
  color?: number[];
  /** Optional display label */
  label?: string | null;
}

/**
 * Response model for force/torque vector data. Postconditions: - vectors list may be empty if no forces computed See issue #1209
 */
export interface ForceVectorResponse {
  /** Current simulation time */
  sim_time: number;
  /** Gravity force vector g(q) */
  gravity_forces?: number[] | null;
  /** Ground reaction forces */
  contact_forces?: number[] | null;
  /** Currently applied torques */
  applied_torques: number[];
  /** Bias forces C(q,v) + g(q) */
  bias_forces?: number[] | null;
}

/**
 * Golfer preset description and initial physical parameters.
 */
export interface GolferPresetInfo {
  /** Human-readable preset name */
  name: string;
  /** Lumped arm mass in kg */
  arm_mass_kg: number;
  /** Equivalent shaft mass in kg */
  shaft_mass_kg: number;
  /** Equivalent head mass in kg */
  clubhead_mass_kg: number;
  /** Arm length hub-to-hands in m */
  arm_length_m: number;
  /** Club length hands-to-head in m */
  club_length_m: number;
  /** Arm angle at top of backswing in rad */
  top_arm_angle_rad: number;
  /** Wrist cock at top of backswing in rad */
  top_wrist_cock_rad: number;
  /** Default downswing duration in s */
  duration_s: number;
  /** Default peak hub torque in N*m */
  hub_torque_nm: number;
  /** Default peak wrist torque in N*m */
  wrist_torque_nm: number;
  /** Default collocation node count */
  node_count: number;
}

/**
 * Response with slope contour data for visualization.
 */
export interface GreenContourResponse {
  width: number;
  height: number;
  grid_x: number[][];
  grid_y: number[][];
  elevations: number[][];
  hole_position: number[];
}

/**
 * Request for green reading between ball and target.
 */
export interface GreenReadingRequest {
  /** Ball X position [m] */
  ball_x: number;
  /** Ball Y position [m] */
  ball_y: number;
  /** Target X position [m] */
  target_x: number;
  /** Target Y position [m] */
  target_y: number;
  /** Green width [m] */
  green_width: number;
  /** Green height [m] */
  green_height: number;
  stimp_rating: number;
}

/**
 * Response with green reading data.
 */
export interface GreenReadingResponse {
  /** Roll model behind the recommended speed (ADR-0045 F1) */
  roll_model: string;
  distance: number;
  total_break: number;
  recommended_speed: number;
  aim_point: number[];
  elevations: number[];
  slopes: number[][];
}

export interface GroupSummaryV1 {
  dimension: "player" | "session" | "club";
  group_value: string;
  estimate: EstimateSummaryV1;
  trust_level: "explicit_user_attested" | "pseudonymous_stable" | "verified_external";
  evidence: string;
}

export interface GroupingDimensionV1 {
  dimension: "player" | "session" | "club";
  column: string;
  trust_level: "explicit_user_attested" | "pseudonymous_stable" | "verified_external";
  evidence: string;
}

export interface HTTPValidationError {
  detail?: ValidationError[];
}

/**
 * One ``swing_sim.ball_flight_trajectory/1`` record to import for overlay. ``record`` is accepted as an opaque JSON object rather than a typed model on purpose: the wire's own shape is validated by the vendored Tools reader (fail-closed), not re-declared here. Passing it through unmodified means every field the reader checks — unknown fields, missing fields, malformed provenance, non-monotone samples — is enforced exactly as the wire defines it, from either flight-model family (issue #9352, ADR-0047).
 */
export interface ImportBallFlightTrajectoryRequest {
  /** A ball_flight_trajectory/1 JSON record produced by either flight-model family (ud.flight_models or swing_sim.flight). */
  record: Record<string, unknown>;
}

/**
 * Response after importing a dataset.
 */
export interface ImportResponse {
  dataset_id?: string | null;
  name: string;
  format: string;
  columns: string[];
  row_count: number;
}

/**
 * An accepted import, in the same shape the page already plots. ``model_name`` and ``trajectory``/``summary`` mirror :class:`BallFlightModelResult` so the page can render an imported curve through the same 3D scene, profile charts, and metrics table as a computed one. ``model_family`` and ``parameter_digest`` carry the provenance the wire mandates; the page always labels an imported curve with both ``model_family`` and ``model_name`` (ADR-0047), never ``model_name`` alone, so it is never confused with a computed curve from the UD registry.
 */
export interface ImportedBallFlightResponse {
  model_name: string;
  model_key: string;
  model_family: string;
  parameter_digest: string;
  source_id: string;
  frame_id: string;
  trajectory: ImportedTrajectorySample[];
  summary: BallFlightSummary;
}

/**
 * Single imported trajectory sample, already in the plot frame.
 */
export interface ImportedTrajectorySample {
  time_s: number;
  position_m: number[];
  velocity_mps?: number[] | null;
}

/**
 * Body for ``POST /capabilities/{name}/install``.
 */
export interface InstallRequest {
  /** Pass --user to pip. Default false: install into the active venv. */
  allow_user_site: boolean;
  /** Return the command we would run without executing it. */
  dry_run: boolean;
  timeout_seconds: number;
}

/**
 * Body for the install endpoint's response.
 */
export interface InstallResponse {
  install: Record<string, unknown>;
  post_install_report?: FeatureReportModel | null;
}

export interface InterpolationV1 {
  lower_distance_yards: number;
  upper_distance_yards: number;
  fraction: number;
}

/**
 * Response model for joint angle display. See issue #1179
 */
export interface JointAngleDisplay {
  /** Joint name */
  joint_name: string;
  /** Joint angle in radians */
  angle_rad: number;
  /** Joint angle in degrees */
  angle_deg: number;
  /** Joint velocity (rad/s) */
  velocity: number;
  /** Applied torque (N*m) */
  torque: number;
}

/**
 * Single joint position and metadata.
 */
export interface JointData {
  name: string;
  /** [x, y, z] position */
  position: number[];
  confidence: number;
  parent?: string | null;
}

/**
 * Response model for a single joint's info.
 */
export interface JointInfoResponse {
  /** Joint index in state vector */
  index: number;
  /** Joint name */
  name: string;
  /** Maximum absolute torque (N*m) */
  torque_limit: number;
  /** Lower position limit (rad) */
  position_limit_lower: number;
  /** Upper position limit (rad) */
  position_limit_upper: number;
  /** Max absolute velocity (rad/s) */
  velocity_limit: number;
  /** Currently applied torque */
  current_torque: number;
}

/**
 * Complete OpenAPI-compatible v2 result envelope.
 */
export interface LaunchMonitorAnalysisResultV2 {
  contract_version: "2.0.0";
  status: "available" | "partial" | "unavailable";
  analysis: Record<string, unknown> | null;
  units: Record<string, MetricUnitsV2>;
  lineage: AnalysisLineageV2;
  missingness: MissingnessV2;
  availability: AvailabilityV2[];
  uncertainty: UncertaintyV2;
  player_identity: PlayerIdentityV2;
  session_identity?: SessionIdentityV2;
  order_evidence?: OrderEvidenceV2;
  vendor_provenance: VendorProvenanceV2[];
  model_provenance: ModelProvenanceV2[];
  claims?: ClaimsV2;
  warnings: string[];
}

/**
 * Launcher manifest as serialized by ``LauncherManifest.to_dict``. ``launcher_csrf_token``/``launcher_csrf_header`` are only present when the manifest is served by the local launcher backend (``local_server.py``), which attaches the local capability token for mutating endpoints.
 */
export interface LauncherManifestResponse {
  /** Manifest schema version */
  version: string;
  /** Manifest description */
  description: string;
  /** Visible tiles */
  tiles: LauncherTileResponse[];
  /** Canonical category -> display label */
  category_labels?: Record<string, string>;
  /** Local launcher capability token (local mode only) */
  launcher_csrf_token?: string | null;
  /** Header name carrying the capability token */
  launcher_csrf_header?: string | null;
}

/**
 * A single launcher tile as serialized by ``LauncherTile.to_dict``.
 */
export interface LauncherTileResponse {
  /** Unique tile identifier */
  id: string;
  /** Display name shown in both launchers */
  name: string;
  /** Brief description under the tile */
  description: string;
  /** Canonical launcher category */
  category: "physics_engine" | "biomechanics" | "simulation" | "motion_matching" | "motion_capture" | "analysis" | "documentation" | "external" | "developer_tools" | "tool";
  /** Engine/handler type for launch dispatch */
  type: string;
  /** Relative path to the script/entry point */
  path: string;
  /** Logo filename relative to assets dir */
  logo: string;
  /** Status chip text (gui_ready, etc.) */
  status: string;
  /** Capability tags */
  capabilities: string[];
  /** Display order (1 = first) */
  order: number;
  /** Engine type for engines */
  engine_type?: string | null;
  /** Provider id for external tiles */
  provider?: string | null;
  /** Provider source root path */
  source_root?: string | null;
  /** Working directory override */
  working_dir?: string | null;
  /** Extra PYTHONPATH roots */
  python_paths?: string[] | null;
  /** URL path for web tools */
  web_route?: string | null;
  /** Web reachability contract (issue #7461) */
  web?: WebLaunchContractResponse | null;
  /** tab | dock | window | external */
  default_launch: string;
  /** Shell surfaces the tile supports (pyqt6, react) */
  shell_surfaces?: string[] | null;
  /** Free-form filter tags */
  tags?: string[] | null;
  /** Hidden legacy-alias tile */
  hidden?: boolean | null;
  /** Why the tile is hidden */
  hidden_reason?: string | null;
  /** Owner of the hidden state */
  hidden_owner?: string | null;
}

/**
 * Login request model.
 */
export interface LoginRequest {
  email: string;
  password: string;
}

/**
 * Login response model.
 */
export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: UserResponse;
}

export interface LongitudinalClaimsV1 {
  association_scope: "descriptive_directional";
  primary_unit: "player_session_stratum";
  shot_level_inference: boolean;
  causal_improvement: boolean;
  confounder_control_is_causal: boolean;
}

export interface LongitudinalDesignV1 {
  primary_unit: "player_session_stratum";
  session_aggregate: "mean" | "median";
  strata: string[];
  confounders: string[];
  pooled_terms: string[];
}

export interface LongitudinalDimensionV1 {
  order_column: string;
  order_unit: string;
  group_column?: string | null;
  group_dimension?: "player" | "session" | "club" | null;
  trust_level: "explicit_user_attested" | "pseudonymous_stable" | "verified_external";
  evidence: string;
  min_samples: number;
  method: "session-cell-sg-trend/1" | "shot-level-sg-trend/1";
}

export interface LongitudinalMissingnessV1 {
  input_row_count: number;
  included_shot_count: number;
  session_cell_count: number;
  excluded_by_reason: Record<string, number>;
}

export interface LongitudinalPlayerAssociationV1 {
  player_id: string;
  session_count: number;
  estimate_per_order_unit?: number | null;
  direction: "increasing" | "decreasing" | "flat" | "unavailable";
  state: "available" | "unavailable";
  reason_code?: string | null;
  standard_error?: number | null;
  ci_lower?: number | null;
  ci_upper?: number | null;
  p_value?: number | null;
  r_squared?: number | null;
  first_to_last_change?: number | null;
}

/**
 * Bounded rows plus attested identity, order, and session-unit design.
 */
export interface LongitudinalSessionPayloadV1 {
  records: Record<string, unknown>[];
  request: LongitudinalSessionRequestV1;
  context: AnalysisContextV2;
}

/**
 * Scientific choices for session-unit directional association.
 */
export interface LongitudinalSessionRequestV1 {
  metric: string;
  direction: "higher_is_better" | "lower_is_better" | "descriptive_only";
  session_aggregate: "mean" | "median";
  strata: string[];
  confounders: string[];
  minimum_sessions_per_player: number;
  minimum_player_clusters: number;
  confidence_level: number;
  pooled_method: "ud-cluster-robust-fe/1" | "dl-random-effects/1";
}

/**
 * Evidence-bearing result that never labels association as improvement.
 */
export interface LongitudinalSessionResultV1 {
  contract_version: "launch-monitor-longitudinal-session/1.0.0";
  status: "available" | "partial" | "unavailable";
  request: LongitudinalSessionRequestV1;
  design: LongitudinalDesignV1;
  session_aggregates: SessionAggregateV1[];
  player_associations: LongitudinalPlayerAssociationV1[];
  pooled_association: PooledAssociationV1 | null;
  availability: AvailabilityV2[];
  missingness: LongitudinalMissingnessV1;
  lineage: AnalysisLineageV2;
  player_identity: PlayerIdentityV2;
  session_identity: SessionIdentityV2;
  order_evidence: OrderEvidenceV2;
  claims?: LongitudinalClaimsV1;
  warnings: string[];
}

export interface LongitudinalSummaryV1 {
  group_dimension: "player" | "session" | "club" | "all";
  group_value: string;
  method: "session-cell-sg-trend/1" | "shot-level-sg-trend/1";
  sample_count: number;
  slope: number;
  intercept: number;
  r_squared: number;
  p_value: number;
  slope_unit: string;
  trust_level: "explicit_user_attested" | "pseudonymous_stable" | "verified_external";
  evidence: string;
}

/**
 * Request model for distance measurement between bodies. See issue #1179
 */
export interface MeasurementRequest {
  /** First body name */
  body_a: string;
  /** Second body name */
  body_b: string;
}

/**
 * Response model for measurement between bodies. See issue #1179
 */
export interface MeasurementResult {
  /** First body name */
  body_a: string;
  /** Second body name */
  body_b: string;
  /** Euclidean distance (m) */
  distance: number;
  /** Position of body A [x, y, z] */
  position_a: number[];
  /** Position of body B [x, y, z] */
  position_b: number[];
  /** Position difference [dx, dy, dz] */
  delta: number[];
}

/**
 * Response model for measurement tools data. See issue #1179
 */
export interface MeasurementToolsResponse {
  /** All joint angle displays */
  joint_angles: JointAngleDisplay[];
  /** Distance measurements */
  measurements?: MeasurementResult[];
}

/**
 * Fixed/random Fisher-z synthesis with heterogeneity diagnostics.
 */
export interface MetaAnalysisSummaryV1 {
  state: "available" | "unavailable";
  reason_code?: "insufficient_eligible_players" | null;
  contributor_count: number;
  total_sample_count: number;
  fixed_effect_r?: number | null;
  fixed_ci_lower?: number | null;
  fixed_ci_upper?: number | null;
  random_effect_r?: number | null;
  random_ci_lower?: number | null;
  random_ci_upper?: number | null;
  tau_squared?: number | null;
  q_statistic?: number | null;
  i_squared_pct?: number | null;
}

export interface MetricUnitsV2 {
  canonical_unit: string;
  display_unit: string;
  authority: "canonical_registry" | "source_declared" | "unknown";
}

export interface MissingnessV2 {
  input_row_count: number;
  complete_row_count: number;
  missing_by_variable: Record<string, number>;
  non_numeric_by_variable: Record<string, number>;
  excluded_by_reason: Record<string, number>;
  policy: "pairwise" | "listwise" | "fail";
}

/**
 * Request model for Frankenstein mode: comparing two models side by side. See issue #1200
 */
export interface ModelCompareRequest {
  /** Path to first model file */
  model_a_path: string;
  /** Path to second model file */
  model_b_path: string;
}

/**
 * Response model for model comparison (Frankenstein mode). See issue #1200
 */
export interface ModelCompareResponse {
  /** First model data */
  model_a: ModelExplorerResponse;
  /** Second model data */
  model_b: ModelExplorerResponse;
  /** Joint names present in both models */
  shared_joints?: string[];
  /** Joints only in model A */
  unique_to_a?: string[];
  /** Joints only in model B */
  unique_to_b?: string[];
}

/**
 * Request model for model explorer operations. See issue #1200
 */
export interface ModelExplorerRequest {
  /** Path to the model file */
  model_path: string;
  /** Joint name to angle (rad) mapping for FK preview */
  joint_values?: Record<string, number> | null;
}

/**
 * Response model for the model explorer view. See issue #1200
 */
export interface ModelExplorerResponse {
  /** Model name */
  model_name: string;
  /** Flat list of tree nodes */
  tree: URDFTreeNode[];
  /** Number of joints */
  joint_count: number;
  /** Number of links */
  link_count: number;
  /** Model format (urdf/mjcf) */
  model_format: string;
  /** Path to the model file */
  file_path: string;
}

/**
 * Response model for listing available models. See issue #1201
 */
export interface ModelListResponse {
  /** List of available models with name and format */
  models: Record<string, string>[];
}

export interface ModelProvenanceV2 {
  model_id: string;
  version: string;
  code_commit?: string | null;
  configuration_sha256?: string | null;
  relationship_to_vendor: "independent_physics" | "vendor_comparable_surrogate" | "vendor_reported_output" | "unknown";
}

/**
 * Toast notification preferences.
 */
export interface NotificationSettings {
  /** Auto-dismiss delay for toasts in milliseconds (500–60000). */
  toast_duration_ms: number;
  /** 'all' shows every toast, 'errors' only errors/warnings, 'silent' suppresses all toasts. */
  verbosity: "all" | "errors" | "silent";
}

/**
 * Evidence defining chronological or ordinal order for longitudinal work.
 */
export interface OrderEvidenceV2 {
  trust_level: "not_provided" | "explicit_user_attested" | "source_reported" | "verified_external" | "untrusted_inferred";
  order_column?: string | null;
  order_kind?: "timestamp" | "ordinal" | "source_sequence" | null;
  unit?: string | null;
  evidence?: string | null;
}

export interface OutcomeProxyClaimsV1 {
  is_strokes_gained: false;
  source_backed: false;
  causal_inference: false;
}

/**
 * Bounded records and explicitly non-SG outcome-proxy request.
 */
export interface OutcomeProxyPayloadV1 {
  records: Record<string, unknown>[];
  request: OutcomeProxyRequestV1;
}

export interface OutcomeProxyRequestV1 {
  carry_column: string;
  lateral_column: string;
  carry_unit: "yd" | "m";
  lateral_unit: "yd" | "m";
  target_distance_yards: number;
  shot_id_column?: string | null;
  confidence_level: number;
  min_samples: number;
}

export interface OutcomeProxyResultV1 {
  contract_version: "launch-monitor-outcome-proxy/1.0.0";
  status: "available" | "partial" | "unavailable";
  metric_name: "expected_proximity_dispersion_proxy";
  unit: "yd";
  value_summary: EstimateSummaryV1;
  row_results: OutcomeProxyRowV1[];
  exclusions: ExclusionSummaryV1;
  formula: string;
  units: Record<string, string>;
  claims?: OutcomeProxyClaimsV1;
  limitations: string[];
}

export interface OutcomeProxyRowV1 {
  source_index: number;
  shot_id?: string | null;
  carry_yards: number;
  lateral_yards: number;
  target_distance_yards: number;
  radial_error_yards: number;
}

/**
 * Request for recording playback control.
 */
export interface PlaybackRequest {
  recording_name: string;
  /** play, pause, stop, seek */
  action: string;
  /** Frame to seek to */
  seek_frame?: number | null;
}

/**
 * Response with current playback state.
 */
export interface PlaybackResponse {
  recording_name: string;
  status: string;
  current_frame: number;
  total_frames: number;
}

/**
 * A player-level estimate and normalized meta-analysis weights.
 */
export interface PlayerAssociationV1 {
  player_id: string;
  estimate: AssociationEstimateV1;
  fixed_weight?: number | null;
  random_weight?: number | null;
}

/**
 * Bounded records, selected pair, and explicit evidence context.
 */
export interface PlayerCovariationPayloadV1 {
  records: Record<string, unknown>[];
  request: PlayerCovariationRequestV1;
  context?: AnalysisContextV2;
}

/**
 * Select one variable pair and the player-level inference rules.
 */
export interface PlayerCovariationRequestV1 {
  x_column: string;
  y_column: string;
  player_column: string;
  min_samples: number;
  confidence_level: number;
}

/**
 * Evidence-bearing result for one selected variable pair.
 */
export interface PlayerCovariationResultV1 {
  contract_version: "launch-monitor-player-covariation/1.0.0";
  analysis_kind: "selected_pair";
  status: "available" | "partial" | "unavailable";
  request: PlayerCovariationRequestV1;
  pooled: AssociationEstimateV1;
  within_player: AssociationEstimateV1;
  between_player: AssociationEstimateV1;
  per_player: PlayerAssociationV1[];
  meta_analysis: MetaAnalysisSummaryV1;
  missingness: CovariationMissingnessV1;
  units: Record<string, MetricUnitsV2>;
  lineage: AnalysisLineageV2;
  availability: AvailabilityV2[];
  uncertainty: CovariationUncertaintyV1;
  player_identity: PlayerIdentityV2;
  vendor_provenance: VendorProvenanceV2[];
  claims?: ClaimsV2;
  definitions: Record<string, string>;
  warnings: string[];
  method_description: string;
}

/**
 * Bounded records and a bounded exploratory pair-scan request.
 */
export interface PlayerCovariationScanPayloadV1 {
  records: Record<string, unknown>[];
  request: PlayerCovariationScanRequestV1;
  context?: AnalysisContextV2;
}

/**
 * Select a bounded exploratory all-pairs scan.
 */
export interface PlayerCovariationScanRequestV1 {
  player_column: string;
  numeric_columns: string[];
  min_samples: number;
  confidence_level: number;
}

/**
 * Evidence-bearing deterministic exploratory pair ranking.
 */
export interface PlayerCovariationScanResultV1 {
  contract_version: "launch-monitor-player-covariation/1.0.0";
  analysis_kind: "pair_scan";
  status: "available" | "partial" | "unavailable";
  request: PlayerCovariationScanRequestV1;
  pair_count: number;
  available_pair_count: number;
  unavailable_pair_count: number;
  ranking: CovariationPairRankV1[];
  lineage: AnalysisLineageV2;
  player_identity: PlayerIdentityV2;
  vendor_provenance: VendorProvenanceV2[];
  claims?: ClaimsV2;
  warnings: string[];
  method_description: string;
}

/**
 * Declared identity evidence; identity is never inferred from layout.
 */
export interface PlayerIdentityV2 {
  trust_level: "not_provided" | "explicit_user_attested" | "pseudonymous_stable" | "verified_external" | "untrusted_inferred";
  /** Explicit player identifier column; session, club, source, file, and row-position fields are forbidden. */
  identifier_column?: string | null;
  evidence?: string | null;
}

/**
 * One enumerable plot type served by the analysis API.
 */
export interface PlotTypeInfo {
  /** Registry id, e.g. 'joint_angles' */
  id: string;
  /** Human-readable dashboard label */
  label: string;
}

/**
 * Catalogue of plot types available from the orchestrator registry.
 */
export interface PlotTypesResponse {
  plot_types: PlotTypeInfo[];
}

/**
 * One pooled estimate, always carrying the name of the estimator. G1-D1: results from different estimators are never numerically compared without the names attached, so ``method`` is required and has no default. ``standard_error``, ``confidence_interval_*`` and ``confidence_level`` are produced by both estimators; ``p_value`` by ``ud-cluster-robust-fe/1``; the heterogeneity block and ``improvement_probability`` by ``dl-random-effects/1``.
 */
export interface PooledAssociationV1 {
  method: "ud-cluster-robust-fe/1" | "dl-random-effects/1";
  estimate_per_order_unit: number;
  standard_error: number;
  confidence_interval_low: number;
  confidence_interval_high: number;
  p_value?: number | null;
  confidence_level: number;
  cluster_count: number;
  session_cell_count: number;
  uncertainty_state: "available";
  tau_squared?: number | null;
  q_statistic?: number | null;
  i_squared_pct?: number | null;
  improvement_probability?: number | null;
}

/**
 * Available golfer presets.
 */
export interface PresetListResponse {
  presets: GolferPresetInfo[];
}

/**
 * Request body for ``POST /realtime/publish``.
 */
export interface PublishRequest {
  /** Channel name (scope/topic pattern) */
  channel: string;
  payload?: Record<string, unknown>;
}

/**
 * Impact quantities displayed beside the slow-motion collision.
 */
export interface Putt3DCollisionResponse {
  ball_speed_mps: number;
  putter_speed_before_mps: number;
  putter_speed_after_mps: number;
  launch_angle_deg: number;
  spin_rad_s: number;
  impulse_n_s: number;
  contact_time_proxy_s: number;
  kinetic_energy_loss_j: number;
  face_twist_rad_s: number;
  twist_moment_n_m_s: number;
}

/**
 * One frame in the three-dimensional playback trajectory.
 */
export interface Putt3DSampleResponse {
  t_s: number;
  x_m: number;
  y_m: number;
  z_m: number;
  speed_mps: number;
  spin_rad_s: number;
  mode: "airborne" | "slide" | "roll" | "rest";
}

/**
 * Physics and visualization inputs for one three-dimensional putt.
 */
export interface Putt3DSimulationRequest {
  putter_speed_mps: number;
  loft_deg: number;
  head_mass_kg: number;
  head_moi_kg_m2: number;
  coefficient_of_restitution: number;
  hosel_toe_m: number;
  hosel_forward_m: number;
  impact_toe_m: number;
  stimp_rating: number;
  grade_percent: number;
  downhill_aspect_deg: number;
  grain_strength: number;
  grain_direction_deg: number;
  rolling_velocity_coefficient: number;
  bump_height_m: number;
  friction_variation: number;
  random_seed: number;
  hole_x_m: number;
  hole_y_m: number;
}

/**
 * Complete deterministic playback payload for the R3F client.
 */
export interface Putt3DSimulationResponse {
  /** Roll model that produced this playback (ADR-0045 F1) */
  roll_model: string;
  samples: Putt3DSampleResponse[];
  collision: Putt3DCollisionResponse;
  surface: Putt3DSurfaceResponse;
  holed: boolean;
  total_distance_m: number;
  duration_s: number;
  skid_distance_m: number;
}

/**
 * Surface geometry metadata required by the R3F scene.
 */
export interface Putt3DSurfaceResponse {
  width_m: number;
  height_m: number;
  grade_percent: number;
  downhill_aspect_deg: number;
  hole_x_m: number;
  hole_y_m: number;
}

/**
 * Request to simulate a single putt.
 */
export interface PuttSimulationRequest {
  /** Ball X position on green [m] */
  ball_x: number;
  /** Ball Y position on green [m] */
  ball_y: number;
  /** Stroke speed [m/s] */
  speed: number;
  /** Aim direction X component */
  direction_x: number;
  /** Aim direction Y component */
  direction_y: number;
  /** Green speed (Stimpmeter) [ft] */
  stimp_rating: number;
  /** Green width [m] */
  green_width: number;
  /** Green height [m] */
  green_height: number;
  /** Hole X position [m] */
  hole_x: number;
  /** Hole Y position [m] */
  hole_y: number;
  /** Wind speed [m/s] */
  wind_speed: number;
  /** Wind direction X */
  wind_direction_x: number;
  /** Wind direction Y */
  wind_direction_y: number;
}

/**
 * Response containing putt simulation results.
 */
export interface PuttSimulationResponse {
  /** Roll model that produced this result (ADR-0045 F1) */
  roll_model: string;
  positions: number[][];
  velocities: number[][];
  times: number[];
  holed: boolean;
  final_position: number[];
  total_distance: number;
  duration: number;
}

/**
 * Metadata about a motion capture recording.
 */
export interface RecordingInfo {
  name: string;
  source_type: string;
  total_frames: number;
  duration_seconds: number;
  frame_rate: number;
  joint_names: string[];
}

/**
 * Request body for token refresh endpoint.
 */
export interface RefreshTokenRequest {
  refresh_token: string;
}

/**
 * Response model for token refresh endpoint.
 */
export interface RefreshTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

/**
 * Request to create or update a custom theme.
 */
export interface SaveCustomThemeRequest {
  /** Custom theme name */
  name: string;
  /** Color key-value pairs */
  colors: Record<string, string>;
  /** Apply theme immediately after saving */
  apply: boolean;
}

/**
 * Request for scatter analysis (multiple putts with variance).
 */
export interface ScatterAnalysisRequest {
  ball_x: number;
  ball_y: number;
  speed: number;
  direction_x: number;
  direction_y: number;
  n_simulations: number;
  speed_variance: number;
  direction_variance_deg: number;
  green_width: number;
  green_height: number;
  stimp_rating: number;
}

/**
 * Response with scatter analysis results.
 */
export interface ScatterAnalysisResponse {
  /** Roll model that produced these results (ADR-0045 F1) */
  roll_model: string;
  final_positions: number[][];
  holed_count: number;
  total_simulations: number;
  average_distance_from_hole: number;
  make_percentage: number;
}

export interface SessionAggregateV1 {
  player_id: string;
  session_id: string;
  order_value: number;
  order_unit: string;
  stratum: Record<string, string>;
  shot_count: number;
  metric_value: number;
  confounder_values: Record<string, number>;
}

/**
 * Evidence for repeated-observation session boundaries, never a player ID.
 */
export interface SessionIdentityV2 {
  trust_level: "not_provided" | "explicit_user_attested" | "source_reported" | "verified_external" | "untrusted_inferred";
  identifier_column?: string | null;
  evidence?: string | null;
}

/**
 * Request to change the active theme.
 */
export interface SetActiveThemeRequest {
  /** Theme name to activate */
  name: string;
}

/**
 * Default simulation parameters applied at app start. Mirrors the desktop Configuration tab (see issue #7457); the web simulation store hydrates from these once per app start so an in-session change is never clobbered (#7424).
 */
export interface SimulationDefaultsSettings {
  /** Engine name preselected on the simulation page. */
  default_engine: string;
  /** Default simulation duration in seconds (0–300]. */
  duration: number;
  /** Default integration timestep in seconds (0–1]. */
  timestep: number;
}

/**
 * Request model for physics simulation. Preconditions: - engine_type must be a known engine identifier - duration must be in (0, 300] seconds - timestep (if given) must be in [1e-6, 0.1] seconds
 */
export interface SimulationRequest {
  /** Physics engine to use (mujoco, drake, etc.) */
  engine_type: string;
  /** Path to model file */
  model_path?: string | null;
  /** Simulation duration in seconds */
  duration: number;
  /** Simulation timestep */
  timestep?: number | null;
  /** Initial joint positions/velocities */
  initial_state?: Record<string, unknown> | null;
  /** Control sequence */
  control_inputs?: Record<string, unknown>[] | null;
  /** Analysis configuration */
  analysis_config?: Record<string, unknown> | null;
}

/**
 * Response model for simulation results. Postconditions: - frames >= 0 - duration >= 0 - data must contain at least 'states' key on success
 */
export interface SimulationResponse {
  /** Whether simulation completed successfully */
  success: boolean;
  /** Actual simulation duration */
  duration: number;
  /** Number of simulation frames */
  frames: number;
  /** Simulation data (states, controls, etc.) */
  data: Record<string, unknown>;
  /** Analysis results if requested */
  analysis_results?: Record<string, unknown> | null;
  /** Paths to exported files */
  export_paths?: string[] | null;
}

/**
 * Response model for simulation runtime statistics. See issue #1202
 */
export interface SimulationStatsResponse {
  /** Current simulation time (s) */
  sim_time: number;
  /** Wall clock time elapsed (s) */
  wall_time: number;
  /** Simulation frames per second */
  fps: number;
  /** Sim time / wall time ratio */
  real_time_factor: number;
  /** Current speed multiplier */
  speed_factor: number;
  /** Whether trajectory is being recorded */
  is_recording: boolean;
  /** Total frames simulated */
  frame_count: number;
}

/**
 * One frame of skeleton data.
 */
export interface SkeletonFrame {
  frame_index: number;
  timestamp: number;
  joints: JointData[];
}

/**
 * Content-addressed source and the sessions it backs.
 */
export interface SourceFileReferenceV2 {
  source_id: string;
  file_sha256: string;
  session_ids: string[];
  source_uri?: string | null;
  rights_status: "public_redistributable" | "restricted_internal" | "permission_required" | "unknown";
}

/**
 * Request model for simulation speed control. Preconditions: - speed_factor must be in [0.1, 10.0] See issue #1202
 */
export interface SpeedControlRequest {
  /** Simulation speed multiplier (0.1x to 10x) */
  speed_factor: number;
}

/**
 * Response model for speed control updates. See issue #1202
 */
export interface SpeedControlResponse {
  /** Applied speed multiplier */
  speed_factor: number;
  /** Status message */
  status: string;
}

export interface StrokesGainedAnalysisResultV1 {
  contract_version: "launch-monitor-strokes-gained-analysis/1.0.0";
  status: "available" | "partial" | "unavailable";
  metric_name: "source_backed_strokes_gained";
  unit: "strokes";
  value_summary: EstimateSummaryV1;
  baseline: BaselineProvenanceV1;
  formula: string;
  units: Record<string, string>;
  availability: AvailabilityV1;
  uncertainty: StrokesGainedUncertaintyV1;
  row_results: StrokesGainedRowV1[];
  excluded_rows: ExcludedRowV1[];
  exclusions: ExclusionSummaryV1;
  group_summaries: GroupSummaryV1[];
  longitudinal_summaries: LongitudinalSummaryV1[];
  analysis_context: AnalysisContextV2;
  dataset_fingerprint_sha256: string;
  claims?: StrokesGainedClaimsV1;
  warnings: string[];
  limitations: string[];
}

export interface StrokesGainedClaimsV1 {
  is_strokes_gained: true;
  source_backed: true;
  device_emulation: false;
  device_certification: false;
  causal_inference: false;
}

/**
 * Bounded records, verified benchmark, and governed SG request.
 */
export interface StrokesGainedPayloadV1 {
  records: Record<string, unknown>[];
  baseline: ExpectedStrokesBaselineV2;
  request: StrokesGainedRequestV1;
  context?: AnalysisContextV2;
}

export interface StrokesGainedRequestV1 {
  start: CourseStateColumnsV1;
  finish: CourseStateColumnsV1;
  shot_id_column?: string | null;
  confidence_level: number;
  min_samples: number;
  summaries: GroupingDimensionV1[];
  longitudinal?: LongitudinalDimensionV1 | null;
}

export interface StrokesGainedRowV1 {
  source_index: number;
  shot_id?: string | null;
  input_record_sha256: string;
  start: CourseStateValueV1;
  finish: CourseStateValueV1;
  expected_start: number;
  expected_finish: number;
  benchmark_standard_error?: number | null;
  strokes_gained: number;
  start_interpolation: InterpolationV1;
  finish_interpolation: InterpolationV1;
  groups?: Record<string, string>;
  longitudinal_order?: number | null;
}

export interface StrokesGainedUncertaintyV1 {
  sampling_method: string;
  confidence_level: number;
  benchmark_method: string;
  benchmark_standard_error_mean?: number | null;
  assumptions: string[];
}

/**
 * Subscription status options.
 */
export type SubscriptionStatus = "active" | "canceled" | "past_due" | "trialing" | "incomplete";

/**
 * Surface material properties.
 */
export interface SurfaceMaterialResponse {
  name: string;
  friction_coefficient: number;
  rolling_resistance: number;
  restitution: number;
  hardness: number;
  grass_height_m: number;
  compressibility: number;
}

/**
 * Wire payload matching comparison schema 1.0.0.
 */
export interface SwingComparisonResponse {
  schema_version: string;
  objective_keys: string[];
  units: Record<string, string>;
  raw_values: Record<string, Record<string, number>>;
  matrix: number[][];
  torque_saturation: Record<string, number[]>;
  swing_distance: number[][];
  is_degenerate: boolean;
  diagnostics: Record<string, Record<string, unknown>>;
}

/**
 * Request model for swing capture import.
 */
export interface SwingImportRequest {
  /** Path to capture file (C3D, CSV, JSON) */
  file_path: string;
  /** Target frame rate for resampling */
  target_frame_rate: number;
  /** Export trajectory for RL training */
  export_for_rl: boolean;
  /** Output path for RL export */
  output_path?: string | null;
}

/**
 * Response model for swing import.
 */
export interface SwingImportResponse {
  status: string;
  n_frames: number;
  n_joints: number;
  duration: number;
  joint_names: string[];
  phases?: Record<string, number> | null;
  rl_export_path?: string | null;
}

/**
 * Request to optimize downswings across competing objectives.
 */
export interface SwingObjectiveCompareRequest {
  /** Optional preset key or name */
  preset_name?: string | null;
  /** Arm mass in kg */
  arm_mass_kg: number;
  /** Equivalent shaft mass in kg */
  shaft_mass_kg?: number | null;
  /** Equivalent head mass in kg */
  clubhead_mass_kg?: number | null;
  /** Arm length in m */
  arm_length_m: number;
  /** Club length in m */
  club_length_m: number;
  /** Top arm angle in rad */
  top_arm_angle_rad: number;
  /** Top wrist cock angle in rad */
  top_wrist_cock_rad: number;
  /** Downswing duration in s */
  duration_s: number;
  /** Max hub torque in N*m */
  hub_torque_nm: number;
  /** Max wrist torque in N*m */
  wrist_torque_nm: number;
  /** Collocation node count */
  node_count: number;
  /** Optional subset of objective keys to compare (minimum 2) */
  objective_keys?: string[] | null;
}

/**
 * Request for querying terrain properties at a point.
 */
export interface TerrainQueryRequest {
  /** X coordinate in meters */
  x: number;
  /** Y coordinate in meters */
  y: number;
}

/**
 * Response with terrain properties at a point.
 */
export interface TerrainQueryResponse {
  x: number;
  y: number;
  elevation: number;
  slope_angle_deg: number;
  terrain_type: string;
  friction: number;
  restitution: number;
  rolling_resistance: number;
}

/**
 * Full theme definition with name and colors.
 */
export interface ThemeDefinition {
  name: string;
  is_builtin: boolean;
  colors: Record<string, string>;
}

/**
 * Response for listing themes.
 */
export interface ThemeListResponse {
  themes: Record<string, ThemeDefinition>;
}

/**
 * Generic response for theme operations.
 */
export interface ThemeOperationResponse {
  success: boolean;
  message: string;
  theme_name?: string | null;
}

/**
 * Request model for trajectory recording control. See issue #1202
 */
export interface TrajectoryRecordRequest {
  /** Recording action: start, stop, or export */
  action: string;
  /** Export format for trajectory data */
  export_format: string;
}

/**
 * Response model for trajectory recording state. See issue #1202
 */
export interface TrajectoryRecordResponse {
  /** Whether recording is active */
  recording: boolean;
  /** Frames recorded so far */
  frame_count: number;
  /** Status message */
  status: string;
  /** Path to exported file */
  export_path?: string | null;
}

/**
 * Versioned transformation applied before analysis.
 */
export interface TransformRecordV2 {
  transform_id: string;
  version: string;
  parameters_sha256: string;
}

/**
 * Descriptor for a single URDF joint. See issue #1201
 */
export interface URDFJointDescriptor {
  /** Joint name */
  name: string;
  /** Joint type: revolute, prismatic, fixed, continuous, floating */
  joint_type: string;
  /** Parent link name */
  parent_link: string;
  /** Child link name */
  child_link: string;
  /** Joint origin [x, y, z] */
  origin?: number[];
  /** Joint orientation [roll, pitch, yaw] */
  rotation?: number[];
  /** Joint rotation axis [x, y, z] */
  axis?: number[];
  /** Lower position limit (rad) */
  lower_limit?: number | null;
  /** Upper position limit (rad) */
  upper_limit?: number | null;
}

/**
 * Geometry descriptor for a single URDF link visual. See issue #1201
 */
export interface URDFLinkGeometry {
  /** Name of the link */
  link_name: string;
  /** Geometry type: box, cylinder, sphere, or mesh */
  geometry_type: string;
  /** Geometry dimensions (size, radius, length, etc.) */
  dimensions?: Record<string, number>;
  /** Translation [x, y, z] */
  origin?: number[];
  /** Rotation [roll, pitch, yaw] in radians */
  rotation?: number[];
  /** RGBA color [r, g, b, a] */
  color?: number[];
  /** Path to mesh file if type=mesh */
  mesh_path?: string | null;
}

/**
 * Response model for parsed URDF model data. Provides all the information needed to render the model in the frontend Three.js scene without direct XML parsing. See issue #1201
 */
export interface URDFModelResponse {
  /** Name of the robot model */
  model_name: string;
  /** Visual geometries for links */
  links: URDFLinkGeometry[];
  /** Joint descriptors */
  joints: URDFJointDescriptor[];
  /** Root link name (no parent) */
  root_link: string;
  /** Raw URDF XML for advanced clients */
  urdf_raw?: string | null;
}

/**
 * A single node in the URDF tree for the model explorer. See issue #1200
 */
export interface URDFTreeNode {
  /** Unique node ID */
  id: string;
  /** Display name */
  name: string;
  /** Node type: link, joint, or root */
  node_type: string;
  /** Parent node ID */
  parent_id?: string | null;
  /** Child node IDs */
  children?: string[];
  /** Node-specific properties */
  properties?: Record<string, unknown>;
}

export interface UncertaintyV2 {
  confidence_level: number;
  correlation_interval: string;
  regression_interval: string;
  multiplicity_adjustment: string;
  assumptions: string[];
}

/**
 * Usage quota item schema.
 */
export interface UsageQuotaItem {
  used: number;
  limit: number;
  remaining: number;
}

/**
 * Response model for usage info endpoint.
 */
export interface UsageSummaryResponse {
  subscription_tier: string;
  api_calls: UsageQuotaItem;
  video_analyses: UsageQuotaItem;
  simulations: UsageQuotaItem;
}

/**
 * User creation model.
 */
export interface UserCreate {
  email: string;
  full_name?: string | null;
  organization?: string | null;
  /** Password must be at least 8 characters */
  password: string;
}

/**
 * User response model (excludes sensitive data).
 */
export interface UserResponse {
  email: string;
  full_name?: string | null;
  organization?: string | null;
  id: number;
  role: UserRole;
  is_active: boolean;
  is_verified: boolean;
  subscription_status: SubscriptionStatus;
  api_calls_this_month: number;
  video_analyses_this_month: number;
  simulations_this_month: number;
  created_at: string;
  last_login?: string | null;
}

/**
 * User roles for role-based access control.
 */
export type UserRole = "free" | "professional" | "enterprise" | "admin";

export interface ValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
  input?: unknown;
  ctx?: Record<string, unknown>;
}

export interface VendorProvenanceV2 {
  vendor: string;
  models: string[];
  software_versions: string[];
  row_count: number;
  metric_statuses: Record<string, string[]>;
}

/**
 * Response model for video analysis results.
 */
export interface VideoAnalysisResponse {
  /** Original filename */
  filename: string;
  /** Total frames in video */
  total_frames: number;
  /** Frames with valid pose detection */
  valid_frames: number;
  /** Average confidence score */
  average_confidence: number;
  /** Quality assessment metrics */
  quality_metrics: Record<string, unknown>;
  /** Pose estimation results */
  pose_data: Record<string, unknown>[];
}

/**
 * How a tile is reachable from the web app (issue #7461). Mirrors ``src.config.launcher_manifest_loader.WebLaunchContract``.
 */
export interface WebLaunchContractResponse {
  /** route | native-window | unavailable */
  mode: string;
  /** In-app route for mode 'route' */
  route?: string | null;
  /** Why unavailable, for mode 'unavailable' */
  reason?: string | null;
}

/**
 * Full per-user web settings document (GET/PUT /settings).
 */
export interface WebSettings {
  appearance?: AppearanceSettings;
  notifications?: NotificationSettings;
  simulation_defaults?: SimulationDefaultsSettings;
}
