"""Simulation service for physics engine management."""

import logging
from typing import Any

from shared.python.dashboard.recorder import GenericPhysicsRecorder
from shared.python.engine_manager import EngineManager
from shared.python.engine_registry import EngineType

from ..models.requests import SimulationRequest
from ..models.responses import SimulationResponse

logger = logging.getLogger(__name__)


class SimulationService:
    """Service for managing physics simulations."""

    def __init__(self, engine_manager: EngineManager):
        """Initialize simulation service.

        Args:
            engine_manager: Engine manager instance
        """
        self.engine_manager = engine_manager
        self.active_simulations: dict[str, Any] = {}

    async def run_simulation(self, request: SimulationRequest) -> SimulationResponse:
        """Run a physics simulation.

        Args:
            request: Simulation request parameters

        Returns:
            Simulation response with results
        """
        try:
            # Load requested engine
            engine_type = EngineType(request.engine_type.upper())
            success = self.engine_manager.load_engine(engine_type)

            if not success:
                return SimulationResponse(
                    success=False,
                    duration=0.0,
                    frames=0,
                    data={"error": f"Failed to load engine: {request.engine_type}"},
                )

            engine = self.engine_manager.get_active_engine()
            if not engine:
                return SimulationResponse(
                    success=False,
                    duration=0.0,
                    frames=0,
                    data={"error": "No active engine available"},
                )

            # Load model if specified
            if request.model_path:
                engine.load_from_path(request.model_path)

            # Set up recorder
            recorder = GenericPhysicsRecorder(engine)
            if request.analysis_config:
                recorder.set_analysis_config(request.analysis_config)

            # Set initial state if provided
            if request.initial_state:
                q = request.initial_state.get("positions", [])
                v = request.initial_state.get("velocities", [])
                if q and v:
                    engine.set_state(q, v)

            # Run simulation
            recorder.start_recording()

            dt = request.timestep or 0.002  # Default timestep
            steps = int(request.duration / dt)

            for step in range(steps):
                # Apply control if provided
                if request.control_inputs and step < len(request.control_inputs):
                    control = request.control_inputs[step]
                    if "torques" in control:
                        engine.set_control(control["torques"])

                # Step simulation
                engine.step(dt)
                recorder.record_step()

            recorder.stop_recording()

            # Extract results
            simulation_data = {
                "time": recorder.get_time_series("time")[1].tolist(),
                "positions": recorder.get_time_series("joint_positions")[1].tolist(),
                "velocities": recorder.get_time_series("joint_velocities")[1].tolist(),
            }

            # Add analysis results if configured
            quality_metrics = {}
            if request.analysis_config:
                if request.analysis_config.get("ztcf"):
                    ztcf_data = recorder.get_time_series("ztcf_accel")
                    if ztcf_data[1] is not None:
                        simulation_data["ztcf_acceleration"] = ztcf_data[1].tolist()

                if request.analysis_config.get("zvcf"):
                    zvcf_data = recorder.get_time_series("zvcf_accel")
                    if zvcf_data[1] is not None:
                        simulation_data["zvcf_acceleration"] = zvcf_data[1].tolist()

            return SimulationResponse(
                success=True,
                duration=request.duration,
                frames=steps,
                data=simulation_data,
                quality_metrics=quality_metrics,
            )

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return SimulationResponse(
                success=False, duration=0.0, frames=0, data={"error": str(e)}
            )

    async def run_simulation_background(
        self, task_id: str, request: SimulationRequest, active_tasks: dict[str, Any]
    ) -> None:
        """Run simulation as background task.

        Args:
            task_id: Unique task identifier
            request: Simulation request
            active_tasks: Dictionary to store task status
        """
        try:
            active_tasks[task_id] = {"status": "running", "progress": 0}

            result = await self.run_simulation(request)

            active_tasks[task_id] = {
                "status": "completed" if result.success else "failed",
                "result": result.dict(),
            }

        except Exception as e:
            active_tasks[task_id] = {"status": "failed", "error": str(e)}
