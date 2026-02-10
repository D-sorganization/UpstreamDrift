"""Request models for Golf Modeling Suite API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

# Valid engine types — keep in sync with EngineType enum
VALID_ENGINE_TYPES = {
    "mujoco",
    "drake",
    "pinocchio",
    "opensim",
    "myosim",
    "myosuite",
    "matlab_2d",
    "matlab_3d",
    "pendulum",
    "putting_green",
}

VALID_ANALYSIS_TYPES = {
    "kinematics",
    "kinetics",
    "muscle",
    "energetics",
    "swing_sequence",
    "center_of_pressure",
    "club_head",
    "joint_torque",
    "power",
    "custom",
}

VALID_EXPORT_FORMATS = {"json", "csv", "mat", "hdf5", "c3d"}

MAX_SIMULATION_DURATION = 300.0  # 5 minutes max
MIN_TIMESTEP = 1e-6
MAX_TIMESTEP = 0.1


class SimulationRequest(BaseModel):
    """Request model for physics simulation.

    Preconditions:
        - engine_type must be a known engine identifier
        - duration must be in (0, 300] seconds
        - timestep (if given) must be in [1e-6, 0.1] seconds
    """

    engine_type: str = Field(
        ..., description="Physics engine to use (mujoco, drake, etc.)"
    )
    model_path: str | None = Field(None, description="Path to model file")
    duration: float = Field(
        1.0,
        description="Simulation duration in seconds",
        gt=0,
        le=MAX_SIMULATION_DURATION,
    )
    timestep: float | None = Field(
        None,
        description="Simulation timestep",
        gt=0,
        le=MAX_TIMESTEP,
    )
    initial_state: dict[str, Any] | None = Field(
        None, description="Initial joint positions/velocities"
    )
    control_inputs: list[dict[str, Any]] | None = Field(
        None, description="Control sequence"
    )
    analysis_config: dict[str, Any] | None = Field(
        None, description="Analysis configuration"
    )

    @field_validator("engine_type")
    @classmethod
    def validate_engine_type(cls, v: str) -> str:
        """Precondition: engine_type must be a recognized engine."""
        normalized = v.lower().strip()
        if normalized not in VALID_ENGINE_TYPES:
            raise ValueError(
                f"Unknown engine_type '{v}'. Valid types: {sorted(VALID_ENGINE_TYPES)}"
            )
        return normalized

    @field_validator("timestep")
    @classmethod
    def validate_timestep_range(cls, v: float | None) -> float | None:
        """Precondition: timestep must be physically reasonable."""
        if v is not None and v < MIN_TIMESTEP:
            raise ValueError(
                f"Timestep {v} is below minimum {MIN_TIMESTEP}. "
                "Sub-microsecond timesteps are not supported."
            )
        return v


class AnalysisRequest(BaseModel):
    """Request model for biomechanical analysis.

    Preconditions:
        - analysis_type must be a known analysis type
        - export_format must be a supported format
    """

    analysis_type: str = Field(
        ..., description="Type of analysis (kinematics, kinetics, etc.)"
    )
    data_source: str = Field(..., description="Source of data (simulation, c3d, video)")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Analysis parameters"
    )
    export_format: str = Field("json", description="Output format")

    @field_validator("analysis_type")
    @classmethod
    def validate_analysis_type(cls, v: str) -> str:
        """Precondition: analysis_type must be recognized."""
        normalized = v.lower().strip()
        if normalized not in VALID_ANALYSIS_TYPES:
            raise ValueError(
                f"Unknown analysis_type '{v}'. "
                f"Valid types: {sorted(VALID_ANALYSIS_TYPES)}"
            )
        return normalized

    @field_validator("export_format")
    @classmethod
    def validate_export_format(cls, v: str) -> str:
        """Precondition: export_format must be supported."""
        normalized = v.lower().strip()
        if normalized not in VALID_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported export_format '{v}'. "
                f"Supported: {sorted(VALID_EXPORT_FORMATS)}"
            )
        return normalized


class VideoAnalysisRequest(BaseModel):
    """Request model for video-based pose estimation."""

    estimator_type: str = Field("mediapipe", description="Pose estimator to use")
    min_confidence: float = Field(
        0.5, description="Minimum confidence threshold", ge=0, le=1
    )
    enable_temporal_smoothing: bool = Field(
        True, description="Enable temporal smoothing"
    )
    max_frames: int | None = Field(None, description="Maximum frames to process")
    export_keypoints: bool = Field(True, description="Export raw keypoints")
    export_joint_angles: bool = Field(True, description="Export computed joint angles")


class ModelFittingRequest(BaseModel):
    """Request model for model fitting to experimental data."""

    model_path: str = Field(..., description="Path to biomechanical model")
    data_path: str = Field(..., description="Path to experimental data (C3D, etc.)")
    fitting_method: str = Field("least_squares", description="Fitting algorithm")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Fitting parameters"
    )


# ──────────────────────────────────────────────────────────────
#  Phase 2: Shared Physics Backend API Requests (#1209, #1202)
# ──────────────────────────────────────────────────────────────

VALID_CONTROL_STRATEGIES = {
    "zero",
    "direct_torque",
    "pd",
    "pid",
    "computed_torque",
    "gravity_compensation",
    "whole_body",
    "impedance",
}

VALID_CAMERA_PRESETS = {"side", "front", "top", "follow_ball", "follow_club"}


class ActuatorUpdateRequest(BaseModel):
    """Request model for per-actuator parameter updates.

    Preconditions:
        - strategy must be a known control strategy
        - torques, if provided, must be a list of floats
        - gains must be positive when provided

    See issue #1209
    """

    strategy: str | None = Field(
        None, description="Control strategy (pd, pid, zero, etc.)"
    )
    torques: list[float] | None = Field(None, description="Per-joint torque values")
    kp: float | list[float] | None = Field(
        None, description="Proportional gain(s)", gt=0
    )
    kd: float | list[float] | None = Field(None, description="Derivative gain(s)", gt=0)
    ki: float | list[float] | None = Field(None, description="Integral gain(s)", ge=0)
    target_positions: list[float] | None = Field(
        None, description="Target joint positions (rad)"
    )
    target_velocities: list[float] | None = Field(
        None, description="Target joint velocities (rad/s)"
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str | None) -> str | None:
        """Precondition: strategy must be a recognized control strategy."""
        if v is None:
            return v
        normalized = v.lower().strip()
        if normalized not in VALID_CONTROL_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{v}'. "
                f"Valid strategies: {sorted(VALID_CONTROL_STRATEGIES)}"
            )
        return normalized


class SpeedControlRequest(BaseModel):
    """Request model for simulation speed control.

    Preconditions:
        - speed_factor must be in [0.1, 10.0]

    See issue #1202
    """

    speed_factor: float = Field(
        1.0,
        description="Simulation speed multiplier (0.1x to 10x)",
        ge=0.1,
        le=10.0,
    )


class CameraPresetRequest(BaseModel):
    """Request model for camera preset selection.

    Preconditions:
        - preset must be a known camera preset

    See issue #1202
    """

    preset: str = Field(
        ..., description="Camera preset (side, front, top, follow_ball, follow_club)"
    )

    @field_validator("preset")
    @classmethod
    def validate_preset(cls, v: str) -> str:
        """Precondition: preset must be a recognized camera preset."""
        normalized = v.lower().strip()
        if normalized not in VALID_CAMERA_PRESETS:
            raise ValueError(
                f"Unknown preset '{v}'. "
                f"Valid presets: {sorted(VALID_CAMERA_PRESETS)}"
            )
        return normalized


class TrajectoryRecordRequest(BaseModel):
    """Request model for trajectory recording control.

    See issue #1202
    """

    action: str = Field(..., description="Recording action: start, stop, or export")
    export_format: str = Field("json", description="Export format for trajectory data")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Precondition: action must be start, stop, or export."""
        normalized = v.lower().strip()
        if normalized not in {"start", "stop", "export"}:
            raise ValueError(
                f"Unknown action '{v}'. Valid actions: start, stop, export"
            )
        return normalized


# ──────────────────────────────────────────────────────────────
#  Phase 3: URDF/MJCF Rendering, Analysis Tools, Simulation Controls
#  (#1201, #1203, #1179)
# ──────────────────────────────────────────────────────────────

VALID_EXPORT_DOWNLOAD_FORMATS = {"csv", "json"}


class DataExportRequest(BaseModel):
    """Request model for data export.

    Preconditions:
        - format must be csv or json

    See issue #1203
    """

    format: str = Field("csv", description="Export format (csv, json)")
    include_metrics: bool = Field(True, description="Include metrics data")
    include_time_series: bool = Field(True, description="Include time series")
    time_range: list[float] | None = Field(
        None, description="Optional time range [start, end] in seconds"
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Precondition: format must be a supported export format."""
        normalized = v.lower().strip()
        if normalized not in VALID_EXPORT_DOWNLOAD_FORMATS:
            raise ValueError(
                f"Unsupported format '{v}'. "
                f"Supported: {sorted(VALID_EXPORT_DOWNLOAD_FORMATS)}"
            )
        return normalized


class BodyPositionUpdateRequest(BaseModel):
    """Request model for updating body position in simulation.

    See issue #1179
    """

    body_name: str = Field(..., description="Name of the body to reposition")
    position: list[float] | None = Field(None, description="New position [x, y, z]")
    rotation: list[float] | None = Field(
        None, description="New rotation [roll, pitch, yaw] in radians"
    )

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: list[float] | None) -> list[float] | None:
        """Precondition: position must have 3 elements."""
        if v is not None and len(v) != 3:
            raise ValueError("Position must be [x, y, z] (3 elements)")
        return v

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, v: list[float] | None) -> list[float] | None:
        """Precondition: rotation must have 3 elements."""
        if v is not None and len(v) != 3:
            raise ValueError("Rotation must be [roll, pitch, yaw] (3 elements)")
        return v


class MeasurementRequest(BaseModel):
    """Request model for distance measurement between bodies.

    See issue #1179
    """

    body_a: str = Field(..., description="First body name")
    body_b: str = Field(..., description="Second body name")
