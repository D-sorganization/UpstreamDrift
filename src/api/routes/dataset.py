"""Dataset generation and export API routes.

Provides endpoints for generating training datasets, importing swing captures,
managing control interfaces, and generating plots.

Design by Contract:
    Preconditions:
        - Server must have engine manager initialized
        - Request bodies must pass Pydantic validation
    Postconditions:
        - Responses contain valid, serializable data
        - Errors return appropriate HTTP status codes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_logger, get_simulation_service

router = APIRouter(prefix="/dataset", tags=["dataset"])


# ---- Request/Response Models ----


class DatasetGenerationRequest(BaseModel):
    """Request model for dataset generation."""

    engine_type: str = Field(..., description="Physics engine to use")
    num_samples: int = Field(10, description="Number of simulation runs", ge=1, le=10000)
    duration: float = Field(2.0, description="Duration per simulation (seconds)", gt=0, le=60)
    timestep: float = Field(0.002, description="Simulation timestep", gt=0, le=0.1)
    seed: int = Field(42, description="Random seed for reproducibility")
    vary_positions: bool = Field(True, description="Randomize initial positions")
    vary_velocities: bool = Field(False, description="Randomize initial velocities")
    record_mass_matrix: bool = Field(True, description="Record inertia matrices")
    record_dynamics: bool = Field(True, description="Record bias/gravity forces")
    record_drift_control: bool = Field(True, description="Record drift/control decomposition")
    export_format: str = Field("hdf5", description="Export format (hdf5, sqlite, csv)")
    output_path: str = Field("output/training_data", description="Output path")


class DatasetGenerationResponse(BaseModel):
    """Response model for dataset generation."""

    status: str
    num_samples: int
    total_frames: int
    export_path: str
    export_format: str


class SwingImportRequest(BaseModel):
    """Request model for swing capture import."""

    file_path: str = Field(..., description="Path to capture file (C3D, CSV, JSON)")
    target_frame_rate: float = Field(200.0, description="Target frame rate for resampling")
    export_for_rl: bool = Field(True, description="Export trajectory for RL training")
    output_path: str | None = Field(None, description="Output path for RL export")


class SwingImportResponse(BaseModel):
    """Response model for swing import."""

    status: str
    n_frames: int
    n_joints: int
    duration: float
    joint_names: list[str]
    phases: dict[str, int] | None = None
    rl_export_path: str | None = None


class ControlStateRequest(BaseModel):
    """Request model for setting control state."""

    strategy: str = Field("zero", description="Control strategy")
    torques: list[float] | None = Field(None, description="Direct torque values")
    joint_index: int | None = Field(None, description="Joint index for single-joint control")
    joint_torque: float | None = Field(None, description="Torque for single joint")
    kp: float | None = Field(None, description="Proportional gain")
    kd: float | None = Field(None, description="Derivative gain")
    ki: float | None = Field(None, description="Integral gain")
    target_positions: list[float] | None = Field(None, description="Target positions")
    target_velocities: list[float] | None = Field(None, description="Target velocities")


class PlotGenerationRequest(BaseModel):
    """Request model for plot generation."""

    plot_types: list[str] | None = Field(None, description="Plot types to generate (None = all)")
    output_dir: str = Field("output/plots", description="Output directory for plots")
    output_format: str = Field("png", description="Image format (png, svg, pdf)")
    dpi: int = Field(150, description="Resolution in DPI")
    joint_indices: list[int] | None = Field(None, description="Specific joints to plot")


class FeatureExecuteRequest(BaseModel):
    """Request model for executing a control feature."""

    feature_name: str = Field(..., description="Feature method name")
    args: dict[str, Any] = Field(default_factory=dict, description="Feature arguments")


# ---- Endpoints ----


@router.post("/generate", response_model=DatasetGenerationResponse)
async def generate_dataset(
    request: DatasetGenerationRequest,
    logger: Any = Depends(get_logger),
) -> DatasetGenerationResponse:
    """Generate a training dataset from simulation runs.

    Varies inputs across simulations and records all kinematics, kinetics,
    and model data for neural network training.
    """
    try:
        from src.shared.python.dataset_generator import (
            ControlProfile,
            DatasetGenerator,
            GeneratorConfig,
        )
        from src.shared.python.mock_engine import MockPhysicsEngine

        # Use mock engine for now (will be replaced with real engine loading)
        engine = MockPhysicsEngine()
        engine.load_from_string("<mock/>")

        config = GeneratorConfig(
            num_samples=request.num_samples,
            duration=request.duration,
            timestep=request.timestep,
            seed=request.seed,
            vary_initial_positions=request.vary_positions,
            vary_initial_velocities=request.vary_velocities,
            record_mass_matrix=request.record_mass_matrix,
            record_bias_forces=request.record_dynamics,
            record_gravity=request.record_dynamics,
            record_drift_control=request.record_drift_control,
            control_profiles=[
                ControlProfile(name="zero"),
                ControlProfile(name="random", profile_type="random", parameters={"scale": 0.5}),
            ],
        )

        generator = DatasetGenerator(engine)
        dataset = generator.generate(config)

        output_path = generator.export(
            dataset, request.output_path, format=request.export_format
        )

        return DatasetGenerationResponse(
            status="success",
            num_samples=dataset.num_samples,
            total_frames=dataset.total_frames,
            export_path=str(output_path),
            export_format=request.export_format,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/import-swing", response_model=SwingImportResponse)
async def import_swing_capture(
    request: SwingImportRequest,
    logger: Any = Depends(get_logger),
) -> SwingImportResponse:
    """Import a golf swing capture file for RL training."""
    try:
        from src.shared.python.swing_capture_import import SwingCaptureImporter

        importer = SwingCaptureImporter(target_frame_rate=request.target_frame_rate)
        trajectory = importer.import_file(request.file_path)

        phases = None
        try:
            phase_labels = importer.detect_swing_phases(trajectory)
            phases = {
                "address": phase_labels.address,
                "backswing_start": phase_labels.backswing_start,
                "top_of_backswing": phase_labels.top_of_backswing,
                "downswing_start": phase_labels.downswing_start,
                "impact": phase_labels.impact,
                "follow_through_end": phase_labels.follow_through_end,
            }
        except Exception:
            pass

        rl_export_path = None
        if request.export_for_rl:
            output = request.output_path or f"output/rl_trajectories/{Path(request.file_path).stem}.json"
            rl_export_path = str(importer.export_for_rl(trajectory, output))

        return SwingImportResponse(
            status="success",
            n_frames=trajectory.n_frames,
            n_joints=trajectory.n_joints,
            duration=float(trajectory.times[-1] - trajectory.times[0]),
            joint_names=trajectory.joint_names,
            phases=phases,
            rl_export_path=rl_export_path,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/control/state")
async def get_control_state(
    logger: Any = Depends(get_logger),
) -> dict[str, Any]:
    """Get current control interface state.

    Returns all joint torques, control strategy, gains, and joint info.
    """
    try:
        from src.shared.python.control_interface import ControlInterface
        from src.shared.python.mock_engine import MockPhysicsEngine

        engine = MockPhysicsEngine()
        engine.load_from_string("<mock/>")
        ctrl = ControlInterface(engine)
        return ctrl.get_state()

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/control/configure")
async def configure_control(
    request: ControlStateRequest,
    logger: Any = Depends(get_logger),
) -> dict[str, Any]:
    """Configure control strategy and parameters."""
    try:
        from src.shared.python.control_interface import ControlInterface
        from src.shared.python.mock_engine import MockPhysicsEngine

        engine = MockPhysicsEngine()
        engine.load_from_string("<mock/>")
        ctrl = ControlInterface(engine)

        if request.strategy:
            ctrl.set_strategy(request.strategy)

        if request.kp is not None or request.kd is not None:
            ctrl.set_gains(kp=request.kp, kd=request.kd, ki=request.ki)

        if request.torques is not None:
            ctrl.set_torques(request.torques)

        if request.joint_index is not None and request.joint_torque is not None:
            ctrl.set_joint_torque(request.joint_index, request.joint_torque)

        if request.target_positions is not None:
            ctrl.set_target_positions(request.target_positions)

        if request.target_velocities is not None:
            ctrl.set_target_velocities(request.target_velocities)

        return ctrl.get_state()

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/control/strategies")
async def get_control_strategies() -> list[dict[str, str]]:
    """Get available control strategies with descriptions."""
    from src.shared.python.control_interface import ControlInterface
    from src.shared.python.mock_engine import MockPhysicsEngine

    engine = MockPhysicsEngine()
    engine.load_from_string("<mock/>")
    ctrl = ControlInterface(engine)
    return ctrl.get_available_strategies()


@router.get("/features")
async def list_features(
    category: str | None = None,
    available_only: bool = False,
) -> list[dict[str, Any]]:
    """List all control and analysis features.

    Exposes all hidden engine capabilities for discoverability.
    """
    try:
        from src.shared.python.control_features_registry import ControlFeaturesRegistry
        from src.shared.python.mock_engine import MockPhysicsEngine

        engine = MockPhysicsEngine()
        engine.load_from_string("<mock/>")
        registry = ControlFeaturesRegistry(engine)
        return registry.list_features(category=category, available_only=available_only)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/features/summary")
async def features_summary() -> dict[str, Any]:
    """Get summary of all available features."""
    try:
        from src.shared.python.control_features_registry import ControlFeaturesRegistry
        from src.shared.python.mock_engine import MockPhysicsEngine

        engine = MockPhysicsEngine()
        engine.load_from_string("<mock/>")
        registry = ControlFeaturesRegistry(engine)
        return registry.get_summary()

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/features/execute")
async def execute_feature(
    request: FeatureExecuteRequest,
    logger: Any = Depends(get_logger),
) -> dict[str, Any]:
    """Execute a specific engine feature by name."""
    try:
        from src.shared.python.control_features_registry import ControlFeaturesRegistry
        from src.shared.python.mock_engine import MockPhysicsEngine

        engine = MockPhysicsEngine()
        engine.load_from_string("<mock/>")
        registry = ControlFeaturesRegistry(engine)
        result = registry.execute(request.feature_name, **request.args)
        return {"feature": request.feature_name, "result": result}

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/plots/types")
async def get_plot_types() -> list[dict[str, str]]:
    """Get available plot types."""
    from src.shared.python.plot_generator import PlotGenerator
    gen = PlotGenerator()
    return gen.get_available_plot_types()


@router.get("/export/formats")
async def get_export_formats() -> list[dict[str, str]]:
    """Get supported export formats."""
    return [
        {"format": "hdf5", "description": "HDF5 hierarchical data (recommended for large datasets)"},
        {"format": "sqlite", "description": "SQLite database (queryable, structured)"},
        {"format": "csv", "description": "CSV files (one per sample, human-readable)"},
        {"format": "mat", "description": "MATLAB .mat files (scipy required)"},
        {"format": "json", "description": "JSON (small datasets, configuration)"},
        {"format": "c3d", "description": "C3D motion capture format (ezc3d required)"},
    ]
