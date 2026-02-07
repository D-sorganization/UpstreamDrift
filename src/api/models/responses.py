"""Response models for Golf Modeling Suite API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class EngineStatusResponse(BaseModel):
    """Response model for engine status."""

    # Frontend-expected fields
    name: str = Field(..., description="Engine name identifier")
    available: bool = Field(..., description="Whether engine is available")
    loaded: bool = Field(False, description="Whether engine is currently loaded")
    version: str | None = Field(None, description="Engine version if available")
    capabilities: list[str] = Field(
        default_factory=list, description="Engine capabilities"
    )

    # Keep for backward compatibility
    engine_type: str = Field(..., description="Engine type identifier (deprecated)")
    status: str = Field(..., description="Current status")
    is_available: bool = Field(
        ..., description="Whether engine is available (deprecated)"
    )
    description: str = Field("", description="Engine description")


class SimulationResponse(BaseModel):
    """Response model for simulation results.

    Postconditions:
        - frames >= 0
        - duration >= 0
        - data must contain at least 'states' key on success
    """

    success: bool = Field(..., description="Whether simulation completed successfully")
    duration: float = Field(..., description="Actual simulation duration", ge=0)
    frames: int = Field(..., description="Number of simulation frames", ge=0)
    data: dict[str, Any] = Field(
        ..., description="Simulation data (states, controls, etc.)"
    )
    analysis_results: dict[str, Any] | None = Field(
        None, description="Analysis results if requested"
    )
    export_paths: list[str] | None = Field(None, description="Paths to exported files")

    @model_validator(mode="after")
    def check_data_on_success(self) -> SimulationResponse:
        """Postcondition: successful simulations must include state data."""
        if self.success and not self.data:
            raise ValueError(
                "Successful simulation must include non-empty data"
            )
        return self


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
    """Response model for biomechanical analysis.

    Postconditions:
        - analysis_type must match the original request
        - results must be non-empty on success
    """

    analysis_type: str = Field(..., description="Type of analysis performed")
    success: bool = Field(..., description="Whether analysis completed successfully")
    results: dict[str, Any] = Field(..., description="Analysis results")
    visualizations: list[str] | None = Field(
        None, description="Generated visualization files"
    )
    export_path: str | None = Field(None, description="Path to exported results")

    @model_validator(mode="after")
    def check_results_on_success(self) -> AnalysisResponse:
        """Postcondition: successful analysis must include results."""
        if self.success and not self.results:
            raise ValueError(
                "Successful analysis must include non-empty results"
            )
        return self


class TaskStatusResponse(BaseModel):
    """Response model for asynchronous task status."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Current task status")
    progress: float | None = Field(None, description="Progress percentage (0-100)")
    result: dict[str, Any] | None = Field(None, description="Task result if completed")
    error: str | None = Field(None, description="Error message if failed")
