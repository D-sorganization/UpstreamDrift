"""Response models for Golf Modeling Suite API."""

from typing import Any

from pydantic import BaseModel, Field


class EngineStatusResponse(BaseModel):
    """Response model for engine status."""

    engine_type: str = Field(..., description="Engine type identifier")
    status: str = Field(..., description="Current status")
    is_available: bool = Field(..., description="Whether engine is available")
    description: str = Field(..., description="Engine description")


class SimulationResponse(BaseModel):
    """Response model for simulation results."""

    success: bool = Field(..., description="Whether simulation completed successfully")
    duration: float = Field(..., description="Actual simulation duration")
    frames: int = Field(..., description="Number of simulation frames")
    data: dict[str, Any] = Field(
        ..., description="Simulation data (states, controls, etc.)"
    )
    analysis_results: dict[str, Any] | None = Field(
        None, description="Analysis results if requested"
    )
    export_paths: list[str] | None = Field(None, description="Paths to exported files")


class VideoAnalysisResponse(BaseModel):
    """Response model for video analysis results."""

    filename: str = Field(..., description="Original filename")
    total_frames: int = Field(..., description="Total frames in video")
    valid_frames: int = Field(..., description="Frames with valid pose detection")
    average_confidence: float = Field(..., description="Average confidence score")
    quality_metrics: dict[str, Any] = Field(
        ..., description="Quality assessment metrics"
    )
    pose_data: list[dict[str, Any]] = Field(..., description="Pose estimation results")


class AnalysisResponse(BaseModel):
    """Response model for biomechanical analysis."""

    analysis_type: str = Field(..., description="Type of analysis performed")
    success: bool = Field(..., description="Whether analysis completed successfully")
    results: dict[str, Any] = Field(..., description="Analysis results")
    visualizations: list[str] | None = Field(
        None, description="Generated visualization files"
    )
    export_path: str | None = Field(None, description="Path to exported results")


class TaskStatusResponse(BaseModel):
    """Response model for asynchronous task status."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Current task status")
    progress: float | None = Field(None, description="Progress percentage (0-100)")
    result: dict[str, Any] | None = Field(None, description="Task result if completed")
    error: str | None = Field(None, description="Error message if failed")
