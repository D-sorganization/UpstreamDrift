"""Simulation service for Golf Modeling Suite API."""

from src.shared.python.logging_config import get_logger
from typing import Any

from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.engine_manager import EngineManager
from src.shared.python.engine_registry import EngineType

from ..models.requests import SimulationRequest
from ..models.responses import SimulationResponse

logger = get_logger(__name__)


class SimulationService:
    """Service for managing physics simulations."""

    def __init__(self, engine_manager: EngineManager):
        """Initialize simulation service.

        Args:
            engine_manager: Engine manager instance
        """
        self.engine_manager = engine_manager

    async def run_simulation(self, request: SimulationRequest) -> SimulationResponse:
        """Run a physics simulation based on request parameters.

        Args:
            request: Simulation request parameters

        Returns:
            Simulation results and data
        """
        try:
            # Load requested engine
            engine_type = EngineType(request.engine_type.upper())
            self.engine_manager._load_engine(
                engine_type
            )  # This method doesn't return success status

            engine = self.engine_manager.get_active_physics_engine()
            if not engine:
                raise RuntimeError(f"Failed to load engine: {request.engine_type}")

            # Load model if specified
            if request.model_path:
                engine.load_from_path(request.model_path)

            # Set initial state if provided
            if request.initial_state:
                q = request.initial_state.get("positions", [])
                v = request.initial_state.get("velocities", [])
                if q and v:
                    engine.set_state(q, v)

            # Setup recorder
            recorder = GenericPhysicsRecorder(engine)

            # Configure analysis if requested
            if request.analysis_config:
                recorder.set_analysis_config(request.analysis_config)

            # Run simulation
            timestep = request.timestep or 0.001
            steps = int(request.duration / timestep)

            # Start recording (using is_recording to check state)
            if not recorder.is_recording:
                recorder.record_step()  # Start recording

            for step in range(steps):
                # Apply control inputs if provided
                if request.control_inputs and step < len(request.control_inputs):
                    control = request.control_inputs[step]
                    if "torques" in control:
                        engine.set_control(control["torques"])

                # Step simulation
                engine.step(timestep)
                recorder.record_step()

            # Extract recorded data
            simulation_data = self._extract_simulation_data(recorder)

            # Perform analysis if requested
            analysis_results = None
            if request.analysis_config:
                analysis_results = self._perform_analysis(
                    recorder, request.analysis_config
                )

            return SimulationResponse(
                success=True,
                duration=request.duration,
                frames=steps,
                data=simulation_data,
                analysis_results=analysis_results,
                export_paths=[],  # Add required field
            )

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return SimulationResponse(
                success=False,
                duration=0.0,
                frames=0,
                data={},
                analysis_results=None,
                export_paths=[],  # Add required field
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

            active_tasks[task_id] = {"status": "completed", "result": result.dict()}

        except Exception as e:
            active_tasks[task_id] = {"status": "failed", "error": str(e)}

    def _extract_simulation_data(
        self, recorder: GenericPhysicsRecorder
    ) -> dict[str, Any]:
        """Extract simulation data from recorder.

        Args:
            recorder: Physics recorder with simulation data

        Returns:
            Dictionary containing simulation data
        """
        data = {}

        try:
            # Extract time series data
            times, positions = recorder.get_time_series("joint_positions")
            data["times"] = times.tolist() if hasattr(times, "tolist") else times
            data["joint_positions"] = (
                positions.tolist() if hasattr(positions, "tolist") else positions
            )

            times, velocities = recorder.get_time_series("joint_velocities")
            data["joint_velocities"] = (
                velocities.tolist() if hasattr(velocities, "tolist") else velocities
            )

            times, accelerations = recorder.get_time_series("joint_accelerations")
            data["joint_accelerations"] = (
                accelerations.tolist()
                if hasattr(accelerations, "tolist")
                else accelerations
            )

            # Extract control data if available
            try:
                times, controls = recorder.get_time_series("control_inputs")
                data["control_inputs"] = (
                    controls.tolist() if hasattr(controls, "tolist") else controls
                )
            except Exception as e:
                logger.debug("Control inputs not available: %s", e)

        except Exception as e:
            logger.warning(f"Error extracting simulation data: {e}")

        return data

    def _perform_analysis(
        self, recorder: GenericPhysicsRecorder, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform analysis on simulation data.

        Args:
            recorder: Physics recorder with simulation data
            config: Analysis configuration

        Returns:
            Analysis results
        """
        results = {}

        try:
            # Extract ZTCF data if enabled
            if config.get("ztcf", False):
                times, ztcf = recorder.get_time_series("ztcf_accel")
                results["ztcf_acceleration"] = (
                    ztcf.tolist() if hasattr(ztcf, "tolist") else ztcf
                )

            # Extract ZVCF data if enabled
            if config.get("zvcf", False):
                times, zvcf = recorder.get_time_series("zvcf_accel")
                results["zvcf_acceleration"] = (
                    zvcf.tolist() if hasattr(zvcf, "tolist") else zvcf
                )

            # Extract drift analysis if enabled
            if config.get("track_drift", False):
                times, drift = recorder.get_time_series("drift_accel")
                results["drift_acceleration"] = (
                    drift.tolist() if hasattr(drift, "tolist") else drift
                )

        except Exception as e:
            logger.warning(f"Error performing analysis: {e}")

        return results
