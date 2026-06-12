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
  /** Maximum simulation time [s] */
  max_time_s: number;
  /** Returned trajectory sample interval [s] */
  time_step_s: number;
}

/**
 * Response containing ball-flight trajectory and summary metrics.
 */
export interface BallFlightSimulationResponse {
  model_name: string;
  model_key: FlightModelType;
  trajectory: BallFlightTrajectorySample[];
  summary: BallFlightSummary;
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
 * Response listing available datasets.
 */
export interface DatasetListResponse {
  datasets: DatasetInfo[];
  total: number;
  search_dir: string;
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
  distance: number;
  total_break: number;
  recommended_speed: number;
  aim_point: number[];
  elevations: number[];
  slopes: number[][];
}

export interface HTTPValidationError {
  detail?: ValidationError[];
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
 * Request body for ``POST /realtime/publish``.
 */
export interface PublishRequest {
  /** Channel name (scope/topic pattern) */
  channel: string;
  payload?: Record<string, unknown>;
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
  final_positions: number[][];
  holed_count: number;
  total_simulations: number;
  average_distance_from_hole: number;
  make_percentage: number;
}

/**
 * Request to change the active theme.
 */
export interface SetActiveThemeRequest {
  /** Theme name to activate */
  name: string;
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
