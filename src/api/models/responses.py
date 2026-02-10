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
            raise ValueError("Successful simulation must include non-empty data")
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
            raise ValueError("Successful analysis must include non-empty results")
        return self


class TaskStatusResponse(BaseModel):
    """Response model for asynchronous task status."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Current task status")
    progress: float | None = Field(None, description="Progress percentage (0-100)")
    result: dict[str, Any] | None = Field(None, description="Task result if completed")
    error: str | None = Field(None, description="Error message if failed")


# ──────────────────────────────────────────────────────────────
#  Phase 2: Shared Physics Backend API Responses (#1209, #1204, #1202)
# ──────────────────────────────────────────────────────────────


class JointInfoResponse(BaseModel):
    """Response model for a single joint's info."""

    index: int = Field(..., description="Joint index in state vector")
    name: str = Field(..., description="Joint name")
    torque_limit: float = Field(..., description="Maximum absolute torque (N*m)")
    position_limit_lower: float = Field(..., description="Lower position limit (rad)")
    position_limit_upper: float = Field(..., description="Upper position limit (rad)")
    velocity_limit: float = Field(..., description="Max absolute velocity (rad/s)")
    current_torque: float = Field(..., description="Currently applied torque")


class ActuatorStateResponse(BaseModel):
    """Response model for actuator/control state.

    See issue #1209
    """

    strategy: str = Field(..., description="Active control strategy")
    n_joints: int = Field(..., description="Number of actuated joints", ge=0)
    joint_names: list[str] = Field(..., description="Names of all joints")
    torques: list[float] = Field(..., description="Current applied torques")
    target_positions: list[float] | None = Field(
        None, description="Target positions if set"
    )
    target_velocities: list[float] | None = Field(
        None, description="Target velocities if set"
    )
    kp: list[float] = Field(..., description="Proportional gains")
    kd: list[float] = Field(..., description="Derivative gains")
    ki: list[float] = Field(..., description="Integral gains")
    joints: list[JointInfoResponse] = Field(..., description="Per-joint details")
    available_strategies: list[dict[str, str]] = Field(
        ..., description="All available control strategies"
    )


class ForceVectorResponse(BaseModel):
    """Response model for force/torque vector data.

    Postconditions:
        - vectors list may be empty if no forces computed

    See issue #1209
    """

    sim_time: float = Field(..., description="Current simulation time")
    gravity_forces: list[float] | None = Field(
        None, description="Gravity force vector g(q)"
    )
    contact_forces: list[float] | None = Field(
        None, description="Ground reaction forces"
    )
    applied_torques: list[float] = Field(..., description="Currently applied torques")
    bias_forces: list[float] | None = Field(
        None, description="Bias forces C(q,v) + g(q)"
    )


class BiomechanicsMetricsResponse(BaseModel):
    """Response model for biomechanics metrics.

    See issue #1209
    """

    sim_time: float = Field(..., description="Current simulation time")
    club_head_speed: float | None = Field(
        None, description="Club head speed (m/s)"
    )
    kinetic_energy: float | None = Field(
        None, description="Total kinetic energy (J)"
    )
    potential_energy: float | None = Field(
        None, description="Total potential energy (J)"
    )
    joint_positions: list[float] = Field(..., description="Current joint positions")
    joint_velocities: list[float] = Field(..., description="Current joint velocities")
    peak_torque: float | None = Field(
        None, description="Peak torque across all joints (N*m)"
    )
    total_torque_magnitude: float | None = Field(
        None, description="Sum of absolute torques (N*m)"
    )


class CapabilityLevelResponse(BaseModel):
    """Response model for a single capability level."""

    name: str = Field(..., description="Capability name")
    level: str = Field(..., description="Support level: full, partial, or none")
    supported: bool = Field(..., description="Whether capability is available")


class EngineCapabilitiesResponse(BaseModel):
    """Response model for engine capabilities.

    See issue #1204
    """

    engine_name: str = Field(..., description="Engine identifier")
    engine_type: str = Field(..., description="Engine type enum value")
    capabilities: list[CapabilityLevelResponse] = Field(
        ..., description="All capabilities with support levels"
    )
    summary: dict[str, int] = Field(
        ..., description="Counts: full, partial, none"
    )


class ControlFeaturesResponse(BaseModel):
    """Response model for control features registry data.

    See issue #1209
    """

    engine: str = Field(..., description="Engine class name")
    total_features: int = Field(..., description="Total registered features")
    available_features: int = Field(..., description="Available on this engine")
    categories: list[dict[str, Any]] = Field(
        ..., description="Feature categories with counts"
    )
    features: list[dict[str, Any]] = Field(..., description="Feature descriptors")


class SimulationStatsResponse(BaseModel):
    """Response model for simulation runtime statistics.

    See issue #1202
    """

    sim_time: float = Field(..., description="Current simulation time (s)")
    wall_time: float = Field(..., description="Wall clock time elapsed (s)")
    fps: float = Field(..., description="Simulation frames per second")
    real_time_factor: float = Field(
        ..., description="Sim time / wall time ratio"
    )
    speed_factor: float = Field(
        ..., description="Current speed multiplier"
    )
    is_recording: bool = Field(..., description="Whether trajectory is being recorded")
    frame_count: int = Field(..., description="Total frames simulated")


class SpeedControlResponse(BaseModel):
    """Response model for speed control updates.

    See issue #1202
    """

    speed_factor: float = Field(..., description="Applied speed multiplier")
    status: str = Field(..., description="Status message")


class CameraPresetResponse(BaseModel):
    """Response model for camera preset application.

    See issue #1202
    """

    preset: str = Field(..., description="Applied camera preset")
    position: list[float] = Field(..., description="Camera position [x, y, z]")
    target: list[float] = Field(..., description="Camera look-at target [x, y, z]")
    up: list[float] = Field(..., description="Camera up vector [x, y, z]")


class TrajectoryRecordResponse(BaseModel):
    """Response model for trajectory recording state.

    See issue #1202
    """

    recording: bool = Field(..., description="Whether recording is active")
    frame_count: int = Field(..., description="Frames recorded so far")
    status: str = Field(..., description="Status message")
    export_path: str | None = Field(None, description="Path to exported file")
