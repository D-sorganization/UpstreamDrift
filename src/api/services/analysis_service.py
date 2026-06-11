"""Analysis service for Golf Modeling Suite API.

This service provides biomechanical analysis capabilities by leveraging
the active physics engine and shared analysis utilities.

Design by Contract:
- Precondition: Engine manager must be initialized
- Postcondition: Returns AnalysisResponse with valid results or error
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.core.error_utils import GolfSuiteError, ValidationError
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.analysis.swing_metrics import SwingMetricsMixin
from src.shared.python.biomechanics.kinematic_sequence import SegmentTimingAnalyzer

from ..models.requests import AnalysisRequest
from ..models.responses import AnalysisResponse

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager

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
        lambda self, request: (
            request.analysis_type is not None and len(request.analysis_type) > 0
        ),
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

    async def _analyze_kinematics(  # noqa: C901
        self, request: AnalysisRequest, engine: Any
    ) -> dict[str, Any]:
        """Perform kinematic analysis (positions, velocities, accelerations).

        Extracts joint kinematics from the physics engine or provided data.
        """
        if not (request is not None):
            raise ValueError("request must be provided")
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

                metadata = result["metadata"]
                metadata["engine_type"] = type(engine).__name__
                metadata["data_source"] = "engine"

            except (GolfSuiteError, ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Could not extract kinematics from engine: %s", e)
                metadata = result["metadata"]
                metadata["engine_error"] = str(e)
                metadata["data_source"] = "none"
        else:
            metadata = result["metadata"]
            metadata["data_source"] = "none"
            metadata["note"] = "No engine loaded - load an engine first"

        # Use provided data if available
        request_data = getattr(request, "data", None)
        if request_data:
            if "joint_angles" in request_data:
                result["joint_angles"] = request_data["joint_angles"]
                result["metadata"]["data_source"] = "request"
            if "angular_velocities" in request_data:
                result["angular_velocities"] = request_data["angular_velocities"]
            if "angular_accelerations" in request_data:
                result["angular_accelerations"] = request_data["angular_accelerations"]

        return result

    async def _analyze_kinetics(  # noqa: C901
        self, request: AnalysisRequest, engine: Any
    ) -> dict[str, Any]:
        """Perform kinetic analysis (forces, torques, moments).

        Extracts joint kinetics from the physics engine or provided data.
        """
        if not (request is not None):
            raise ValueError("request must be provided")
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

                metadata = result["metadata"]
                metadata["engine_type"] = type(engine).__name__
                metadata["data_source"] = "engine"

            except (GolfSuiteError, ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Could not extract kinetics from engine: %s", e)
                metadata = result["metadata"]
                metadata["engine_error"] = str(e)
                metadata["data_source"] = "none"
        else:
            metadata = result["metadata"]
            metadata["data_source"] = "none"
            metadata["note"] = "No engine loaded - load an engine first"

        # Use provided data if available
        request_data = getattr(request, "data", None)
        if request_data and "joint_torques" in request_data:
            result["joint_torques"] = request_data["joint_torques"]
            result["metadata"]["data_source"] = "request"

        return result

    async def _analyze_energetics(  # noqa: C901
        self, request: AnalysisRequest, engine: Any
    ) -> dict[str, Any]:
        """Perform energetic analysis (energy, power, work).

        Computes energy metrics from the physics engine state.
        """
        if not (request is not None):
            raise ValueError("request must be provided")
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

                metadata = result["metadata"]
                metadata["engine_type"] = type(engine).__name__
                metadata["data_source"] = "engine"

            except (GolfSuiteError, ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Could not extract energetics from engine: %s", e)
                metadata = result["metadata"]
                metadata["engine_error"] = str(e)
                metadata["data_source"] = "none"
        else:
            metadata = result["metadata"]
            metadata["data_source"] = "none"
            metadata["note"] = "No engine loaded - load an engine first"

        return result

    async def _analyze_swing_sequence(  # noqa: C901
        self, request: AnalysisRequest, engine: Any
    ) -> dict[str, Any]:
        """Perform swing sequence analysis (phase detection, timing).

        Analyzes the golf swing phases and transitions.
        """
        # Standard golf swing phases
        if not (request is not None):
            raise ValueError("request must be provided")
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
                        self._populate_kinematic_sequence(result, request, seg_vel)

                x_factor = self._compute_x_factor(request, engine)
                if x_factor is not None:
                    result["x_factor"] = x_factor

                metadata = result["metadata"]
                metadata["engine_type"] = type(engine).__name__
                metadata["data_source"] = "engine"

            except (GolfSuiteError, ImportError, TypeError, ValueError) as e:
                logger.warning("Could not analyze swing sequence from engine: %s", e)
                metadata = result["metadata"]
                metadata["engine_error"] = str(e)
                metadata["data_source"] = "none"
        else:
            metadata = result["metadata"]
            metadata["data_source"] = "none"
            metadata["note"] = "No engine loaded - load an engine first"

        # Use provided timing data if available
        request_data = getattr(request, "data", None)
        if request_data:
            if "phase_transitions" in request_data:
                result["phase_transitions"] = request_data["phase_transitions"]
                result["metadata"]["data_source"] = "request"
            if "sequence_timing" in request_data:
                result["sequence_timing"] = request_data["sequence_timing"]

        return result

    def _populate_kinematic_sequence(
        self,
        result: dict[str, Any],
        request: AnalysisRequest,
        segment_velocities: Any,
    ) -> None:
        """Populate computed segment timing or an explicit no-trajectory marker."""
        normalized = self._normalize_segment_velocity_trajectories(segment_velocities)
        metadata = result["metadata"]
        if not normalized:
            metadata["kinematic_sequence"] = "requires_trajectory"
            return

        times = self._resolve_sequence_times(request, normalized)
        if times is None:
            metadata["kinematic_sequence"] = "requires_trajectory"
            return

        expected_order = ["pelvis", "torso", "arm", "club"]
        analyzer = SegmentTimingAnalyzer(expected_order=expected_order)
        timing = analyzer.analyze(normalized, times)

        sequence: dict[str, Any] = {
            "sequence_order": timing.sequence_order,
            "expected_order": timing.expected_order,
            "sequence_consistency": timing.sequence_consistency,
            "is_valid_sequence": timing.is_valid_sequence,
            "timing_gaps": timing.timing_gaps,
        }
        for peak in timing.peaks:
            sequence[f"{peak.name}_peak"] = {
                "velocity": peak.peak_velocity,
                "time": peak.time,
                "index": peak.index,
                "normalized_velocity": peak.normalized_velocity,
            }
            if peak.speed_gain is not None:
                sequence[f"{peak.name}_peak"]["speed_gain"] = peak.speed_gain
            if peak.deceleration_rate is not None:
                sequence[f"{peak.name}_peak"]["deceleration_rate"] = (
                    peak.deceleration_rate
                )

        result["kinematic_sequence"] = sequence
        metadata["kinematic_sequence"] = "computed"

    def _normalize_segment_velocity_trajectories(
        self,
        segment_velocities: Any,
    ) -> dict[str, np.ndarray] | None:
        """Return per-segment 1-D trajectories; reject instantaneous samples."""
        if not isinstance(segment_velocities, dict):
            return None

        normalized: dict[str, np.ndarray] = {}
        trajectory_length: int | None = None
        for name, values in segment_velocities.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size < 2:
                return None
            if trajectory_length is None:
                trajectory_length = arr.size
            elif arr.size != trajectory_length:
                return None
            normalized[str(name)] = arr

        return normalized or None

    def _resolve_sequence_times(
        self,
        request: AnalysisRequest,
        segment_velocities: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        """Resolve a timebase for segment velocity trajectories."""
        trajectory_length = len(next(iter(segment_velocities.values())))
        request_data = getattr(request, "data", None) or {}
        raw_times = request_data.get("times", request_data.get("time"))
        if raw_times is None:
            return np.arange(trajectory_length, dtype=float)

        times = np.asarray(raw_times, dtype=float).reshape(-1)
        if times.size != trajectory_length:
            return None
        return times

    def _compute_x_factor(
        self,
        request: AnalysisRequest,
        engine: Any,
    ) -> dict[str, Any] | None:
        """Compute X-factor only when joint trajectory data and indices are available."""
        request_data = getattr(request, "data", None) or {}
        joint_positions = request_data.get("joint_positions")
        if joint_positions is None and hasattr(engine, "get_joint_positions"):
            joint_positions = engine.get_joint_positions()
        if joint_positions is None:
            return None

        shoulder_idx = request_data.get(
            "shoulder_joint_idx",
            request_data.get("torso_joint_idx"),
        )
        hip_idx = request_data.get(
            "hip_joint_idx", request_data.get("pelvis_joint_idx")
        )
        if shoulder_idx is None or hip_idx is None:
            return None

        positions = np.asarray(joint_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[0] < 2:
            return None

        times = request_data.get("times", request_data.get("time"))
        if times is not None:
            time_array = np.asarray(times, dtype=float).reshape(-1)
            dt = float(np.mean(np.diff(time_array))) if time_array.size > 1 else 1.0
        else:
            dt = 1.0

        class _SwingMetricContext(SwingMetricsMixin):
            pass

        context = _SwingMetricContext()
        context.joint_positions = positions
        context.times = np.arange(positions.shape[0], dtype=float)
        context.club_head_speed = None
        context.dt = dt

        x_factor = context.compute_x_factor(int(shoulder_idx), int(hip_idx))
        if x_factor is None or x_factor.size == 0:
            return None

        stretch = context.compute_x_factor_stretch(int(shoulder_idx), int(hip_idx))
        payload: dict[str, Any] = {
            "values": x_factor.tolist(),
            "peak": float(np.max(np.abs(x_factor))),
        }
        if stretch is not None:
            payload["stretch_rate"] = stretch[0].tolist()
            payload["peak_stretch_rate"] = stretch[1]
        return payload

    def _detect_swing_phase(self, state: dict[str, Any]) -> str | None:
        """Detect current swing phase from engine state.

        Simple heuristic-based phase detection. For production use,
        this should be replaced with ML-based detection.
        """
        if not (state is not None):
            raise ValueError("state must be provided")
        if not state:
            return None

        # Check for common state indicators
        time = state.get("time", 0)
        if time == 0:
            return "address"

        return None  # Unable to determine without more context

    def _to_list(self, data: Any) -> list[Any]:
        """Convert numpy array or other data to JSON-serializable list."""
        if data is None:
            return []
        if isinstance(data, np.ndarray):
            return list(data.tolist())
        if isinstance(data, (list, tuple)):
            return list(data)
        if isinstance(data, int | float):
            return [data]
        return []
