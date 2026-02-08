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
