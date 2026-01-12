"""Request models for Golf Modeling Suite API."""

from typing import Any

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Request for physics simulation."""

    engine_type: str = Field(..., description="Physics engine to use")
    model_path: str | None = Field(None, description="Path to model file")
    duration: float = Field(1.0, description="Simulation duration in seconds")
    timestep: float | None = Field(None, description="Simulation timestep")
    initial_state: dict[str, Any] | None = Field(None, description="Initial conditions")
    control_inputs: list[dict[str, Any]] | None = Field(
        None, description="Control sequence"
    )
    analysis_config: dict[str, Any] | None = Field(
        None, description="Analysis configuration"
    )


class AnalysisRequest(BaseModel):
    """Request for biomechanical analysis."""

    analysis_type: str = Field(..., description="Type of analysis to perform")
    data_source: str = Field(
        ..., description="Source of data (simulation, video, etc.)"
    )
    parameters: dict[str, Any] | None = Field(None, description="Analysis parameters")
    export_format: str = Field("json", description="Output format")


class VideoAnalysisRequest(BaseModel):
    """Request for video-based pose analysis."""

    estimator_type: str = Field("mediapipe", description="Pose estimator to use")
    min_confidence: float = Field(0.5, description="Minimum confidence threshold")
    enable_temporal_smoothing: bool = Field(
        True, description="Enable temporal smoothing"
    )
    max_frames: int | None = Field(None, description="Maximum frames to process")
    export_keypoints: bool = Field(True, description="Export raw keypoints")
    export_joint_angles: bool = Field(True, description="Export computed joint angles")
