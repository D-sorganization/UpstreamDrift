"""Analysis service for Golf Modeling Suite API.

This service provides biomechanical analysis capabilities by leveraging
the active physics engine and shared analysis utilities.

Design by Contract:
- Precondition: Engine manager must be initialized
- Postcondition: Returns AnalysisResponse with valid results or error
"""

from typing import Any

import numpy as np

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.core.error_utils import GolfSuiteError, ValidationError
from src.shared.python.engine_core.engine_manager import EngineManager
from src.shared.python.logging_pkg.logging_config import get_logger

from ..models.requests import AnalysisRequest
from ..models.responses import AnalysisResponse

logger = get_logger(__name__)

VALID_ANALYSIS_TYPES = frozenset(
    {"kinematics", "kinetics", "energetics", "swing_sequence"}
)


class AnalysisService:
    """Service for biomechanical analysis.

    Provides kinematic, kinetic, energetic, and swing sequence analysis
    by interfacing with the active physics engine.
    """

    def __init__(self, engine_manager: EngineManager) -> None:
        """Initialize analysis service.

        Args:
            engine_manager: Engine manager instance for accessing physics engines
        """
        self.engine_manager = engine_manager

    @precondition(
        lambda self, request: request is not None,
        "Analysis request must not be None",
    )
    @precondition(
        lambda self, request: request.analysis_type is not None
        and len(request.analysis_type) > 0,
        "Analysis type must be specified",
    )
    @postcondition(
        lambda result: result.success or "error" in result.results,
        "Must return success or error details",
    )
    async def analyze_biomechanics(self, request: AnalysisRequest) -> AnalysisResponse:
        """Perform biomechanical analysis.

        Args:
            request: Analysis request parameters including analysis_type and data

        Returns:
            AnalysisResponse with computed results or error information

        Raises:
            ValidationError: If analysis_type is not recognized
        """
        # Fail-fast: validate analysis type before doing any work
        if request.analysis_type not in VALID_ANALYSIS_TYPES:
            raise ValidationError(
                field="analysis_type",
                value=request.analysis_type,
                reason="Unknown analysis type",
                valid_values=sorted(VALID_ANALYSIS_TYPES),
            )

        try:
            # Get the active engine for analysis
            engine = self.engine_manager.get_active_physics_engine()

            if request.analysis_type == "kinematics":
                results = await self._analyze_kinematics(request, engine)
            elif request.analysis_type == "kinetics":
                results = await self._analyze_kinetics(request, engine)
            elif request.analysis_type == "energetics":
                results = await self._analyze_energetics(request, engine)
            elif request.analysis_type == "swing_sequence":
                results = await self._analyze_swing_sequence(request, engine)
            else:
                raise ValidationError(
                    field="analysis_type",
                    value=request.analysis_type,
                    reason="Unknown analysis type",
                    valid_values=sorted(VALID_ANALYSIS_TYPES),
                )

            return AnalysisResponse(
                analysis_type=request.analysis_type,
                success=True,
                results=results,
                visualizations=[],
                export_path="",
            )

        except (GolfSuiteError, RuntimeError, OSError) as e:
            logger.error("Analysis failed: %s", e, exc_info=True)
            return AnalysisResponse(
                analysis_type=request.analysis_type,
                success=False,
                results={
                    "error": str(e),
                    "error_code": "GMS-ANL-002",
                    "analysis_type": request.analysis_type,
                },
                visualizations=[],
                export_path="",
            )

    async def _analyze_kinematics(
        self, request: AnalysisRequest, engine: Any
    ) -> dict[str, Any]:
        """Perform kinematic analysis (positions, velocities, accelerations).

        Extracts joint kinematics from the physics engine or provided data.
        """
        result: dict[str, Any] = {
            "analysis_type": "kinematics",
            "joint_angles": [],
            "angular_velocities": [],
            "angular_accelerations": [],
            "metadata": {},
        }

        # Try to get data from active engine
        if engine is not None:
            try:
                # Get joint positions/angles
                if hasattr(engine, "get_joint_positions"):
                    positions = engine.get_joint_positions()
                    if positions is not None:
                        result["joint_angles"] = self._to_list(positions)

                # Get joint velocities
                if hasattr(engine, "get_joint_velocities"):
                    velocities = engine.get_joint_velocities()
                    if velocities is not None:
                        result["angular_velocities"] = self._to_list(velocities)

                # Get joint accelerations
                if hasattr(engine, "get_joint_accelerations"):
                    accelerations = engine.get_joint_accelerations()
                    if accelerations is not None:
                        result["angular_accelerations"] = self._to_list(accelerations)

                # Get state if available for additional data
                if hasattr(engine, "get_state"):
                    state = engine.get_state()
                    if isinstance(state, dict):
                        result["metadata"]["state_keys"] = list(state.keys())

                result["metadata"]["engine_type"] = type(engine).__name__
                result["metadata"]["data_source"] = "engine"

            except (GolfSuiteError, ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Could not extract kinematics from engine: %s", e)
                result["metadata"]["engine_error"] = str(e)
                result["metadata"]["data_source"] = "none"
        else:
            result["metadata"]["data_source"] = "none"
            result["metadata"]["note"] = "No engine loaded - load an engine first"

        # Use provided data if available
        if hasattr(request, "data") and request.data:
            if "joint_angles" in request.data:
                result["joint_angles"] = request.data["joint_angles"]
                result["metadata"]["data_source"] = "request"
            if "angular_velocities" in request.data:
                result["angular_velocities"] = request.data["angular_velocities"]
            if "angular_accelerations" in request.data:
                result["angular_accelerations"] = request.data["angular_accelerations"]

        return result

    async def _analyze_kinetics(
        self, request: AnalysisRequest, engine: Any
    ) -> dict[str, Any]:
        """Perform kinetic analysis (forces, torques, moments).

        Extracts joint kinetics from the physics engine or provided data.
        """
        result: dict[str, Any] = {
            "analysis_type": "kinetics",
            "joint_torques": [],
            "reaction_forces": [],
            "muscle_forces": [],
            "ground_reaction_forces": {},
            "metadata": {},
        }

        if engine is not None:
            try:
                # Get joint torques
                if hasattr(engine, "get_joint_torques"):
                    torques = engine.get_joint_torques()
                    if torques is not None:
                        result["joint_torques"] = self._to_list(torques)

                # Get actuator/muscle forces
                if hasattr(engine, "get_actuator_forces"):
                    forces = engine.get_actuator_forces()
                    if forces is not None:
                        result["muscle_forces"] = self._to_list(forces)

                # Get ground reaction forces if available
                if hasattr(engine, "get_contact_forces"):
                    contact = engine.get_contact_forces()
                    if contact is not None:
                        result["ground_reaction_forces"] = contact

                result["metadata"]["engine_type"] = type(engine).__name__
                result["metadata"]["data_source"] = "engine"

            except (GolfSuiteError, ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Could not extract kinetics from engine: %s", e)
                result["metadata"]["engine_error"] = str(e)
                result["metadata"]["data_source"] = "none"
        else:
            result["metadata"]["data_source"] = "none"
            result["metadata"]["note"] = "No engine loaded - load an engine first"

        # Use provided data if available
        if (
            hasattr(request, "data")
            and request.data
            and "joint_torques" in request.data
        ):
            result["joint_torques"] = request.data["joint_torques"]
            result["metadata"]["data_source"] = "request"

        return result

    async def _analyze_energetics(
        self, request: AnalysisRequest, engine: Any
    ) -> dict[str, Any]:
        """Perform energetic analysis (energy, power, work).

        Computes energy metrics from the physics engine state.
        """
        result: dict[str, Any] = {
            "analysis_type": "energetics",
            "kinetic_energy": 0.0,
            "potential_energy": 0.0,
            "total_energy": 0.0,
            "power": [],
            "energy_flow": {},
            "metadata": {},
        }

        if engine is not None:
            try:
                # Get energy values
                if hasattr(engine, "get_kinetic_energy"):
                    ke = engine.get_kinetic_energy()
                    if ke is not None:
                        result["kinetic_energy"] = float(ke)

                if hasattr(engine, "get_potential_energy"):
                    pe = engine.get_potential_energy()
                    if pe is not None:
                        result["potential_energy"] = float(pe)

                # Calculate total if not provided
                if hasattr(engine, "get_total_energy"):
                    total = engine.get_total_energy()
                    if total is not None:
                        result["total_energy"] = float(total)
                else:
                    result["total_energy"] = (
                        result["kinetic_energy"] + result["potential_energy"]
                    )

                # Get power if available
                if hasattr(engine, "get_actuator_powers"):
                    powers = engine.get_actuator_powers()
                    if powers is not None:
                        result["power"] = self._to_list(powers)

                result["metadata"]["engine_type"] = type(engine).__name__
                result["metadata"]["data_source"] = "engine"

            except (GolfSuiteError, ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Could not extract energetics from engine: %s", e)
                result["metadata"]["engine_error"] = str(e)
                result["metadata"]["data_source"] = "none"
        else:
            result["metadata"]["data_source"] = "none"
            result["metadata"]["note"] = "No engine loaded - load an engine first"

        return result

    async def _analyze_swing_sequence(
        self, request: AnalysisRequest, engine: Any
    ) -> dict[str, Any]:
        """Perform swing sequence analysis (phase detection, timing).

        Analyzes the golf swing phases and transitions.
        """
        # Standard golf swing phases
        SWING_PHASES = [
            "address",
            "takeaway",
            "backswing",
            "transition",
            "downswing",
            "impact",
            "follow_through",
            "finish",
        ]

        result: dict[str, Any] = {
            "analysis_type": "swing_sequence",
            "phases": SWING_PHASES,
            "current_phase": None,
            "phase_transitions": [],
            "sequence_timing": {},
            "kinematic_sequence": {},
            "x_factor": None,
            "metadata": {},
        }

        if engine is not None:
            try:
                # Try to detect current phase from engine state
                if hasattr(engine, "get_state"):
                    state = engine.get_state()
                    if isinstance(state, dict):
                        result["current_phase"] = self._detect_swing_phase(state)

                # Get kinematic sequence data if available
                if hasattr(engine, "get_segment_angular_velocities"):
                    seg_vel = engine.get_segment_angular_velocities()
                    if seg_vel is not None:
                        result["kinematic_sequence"] = {
                            "pelvis_peak": None,
                            "torso_peak": None,
                            "arm_peak": None,
                            "club_peak": None,
                        }

                result["metadata"]["engine_type"] = type(engine).__name__
                result["metadata"]["data_source"] = "engine"

            except (GolfSuiteError, ImportError) as e:
                logger.warning("Could not analyze swing sequence from engine: %s", e)
                result["metadata"]["engine_error"] = str(e)
                result["metadata"]["data_source"] = "none"
        else:
            result["metadata"]["data_source"] = "none"
            result["metadata"]["note"] = "No engine loaded - load an engine first"

        # Use provided timing data if available
        if hasattr(request, "data") and request.data:
            if "phase_transitions" in request.data:
                result["phase_transitions"] = request.data["phase_transitions"]
                result["metadata"]["data_source"] = "request"
            if "sequence_timing" in request.data:
                result["sequence_timing"] = request.data["sequence_timing"]

        return result

    def _detect_swing_phase(self, state: dict) -> str | None:
        """Detect current swing phase from engine state.

        Simple heuristic-based phase detection. For production use,
        this should be replaced with ML-based detection.
        """
        if not state:
            return None

        # Check for common state indicators
        time = state.get("time", 0)
        if time == 0:
            return "address"

        return None  # Unable to determine without more context

    def _to_list(self, data: Any) -> list:
        """Convert numpy array or other data to JSON-serializable list."""
        if data is None:
            return []
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, (list, tuple)):
            return list(data)
        if isinstance(data, (int, float)):
            return [data]
        return []
