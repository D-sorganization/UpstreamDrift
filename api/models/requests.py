"""Request models for Golf Modeling Suite API."""

from typing import Any

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Request model for physics simulation."""

    engine_type: str = Field(
        ..., description="Physics engine to use (mujoco, drake, etc.)"
    )
    model_path: str | None = Field(None, description="Path to model file")
    duration: float = Field(1.0, description="Simulation duration in seconds", gt=0)
    timestep: float | None = Field(None, description="Simulation timestep", gt=0)
    initial_state: dict[str, Any] | None = Field(
        None, description="Initial joint positions/velocities"
    )
    control_inputs: list[dict[str, Any]] | None = Field(
        None, description="Control sequence"
    )
    analysis_config: dict[str, Any] | None = Field(
        None, description="Analysis configuration"
    )


class AnalysisRequest(BaseModel):
    """Request model for biomechanical analysis."""

    analysis_type: str = Field(
        ..., description="Type of analysis (kinematics, kinetics, etc.)"
    )
    data_source: str = Field(..., description="Source of data (simulation, c3d, video)")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Analysis parameters"
    )
    export_format: str = Field("json", description="Output format")


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
