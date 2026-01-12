"""Response models for Golf Modeling Suite API."""

from typing import Any

from pydantic import BaseModel, Field


class EngineStatusResponse(BaseModel):
    """Response for engine status."""

    engine_type: str = Field(..., description="Engine type identifier")
    status: str = Field(..., description="Current status")
    is_available: bool = Field(..., description="Whether engine is available")
    description: str = Field(..., description="Engine description")


class SimulationResponse(BaseModel):
    """Response from physics simulation."""

    success: bool = Field(..., description="Whether simulation succeeded")
    duration: float = Field(..., description="Actual simulation duration")
    frames: int = Field(..., description="Number of simulation frames")
    data: dict[str, Any] | None = Field(None, description="Simulation results")
    quality_metrics: dict[str, Any] | None = Field(None, description="Quality metrics")
    export_paths: list[str] | None = Field(None, description="Paths to exported files")


class VideoAnalysisResponse(BaseModel):
    """Response from video analysis."""

    filename: str = Field(..., description="Original filename")
    total_frames: int = Field(..., description="Total frames in video")
    valid_frames: int = Field(..., description="Frames with valid pose detection")
    average_confidence: float = Field(..., description="Average confidence score")
    quality_metrics: dict[str, Any] = Field(
        ..., description="Quality assessment metrics"
    )
    pose_data: list[dict[str, Any]] = Field(..., description="Pose estimation results")


class AnalysisResponse(BaseModel):
    """Response from biomechanical analysis."""

    analysis_type: str = Field(..., description="Type of analysis performed")
    success: bool = Field(..., description="Whether analysis succeeded")
    results: dict[str, Any] = Field(..., description="Analysis results")
    visualizations: list[str] | None = Field(
        None, description="Generated visualization paths"
    )
    export_paths: list[str] | None = Field(None, description="Exported data paths")
