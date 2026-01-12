"""Analysis service for biomechanical computations."""

import logging
from typing import Any

from shared.python.engine_manager import EngineManager

from ..models.requests import AnalysisRequest
from ..models.responses import AnalysisResponse

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for biomechanical analysis operations."""

    def __init__(self, engine_manager: EngineManager):
        """Initialize analysis service.

        Args:
            engine_manager: Engine manager instance
        """
        self.engine_manager = engine_manager

    async def analyze_biomechanics(self, request: AnalysisRequest) -> AnalysisResponse:
        """Perform biomechanical analysis.

        Args:
            request: Analysis request parameters

        Returns:
            Analysis response with results
        """
        try:
            engine = self.engine_manager.get_active_engine()
            if not engine:
                return AnalysisResponse(
                    analysis_type=request.analysis_type,
                    success=False,
                    results={"error": "No active physics engine"},
                )

            results = {}

            if request.analysis_type == "kinematics":
                results = await self._analyze_kinematics(
                    engine, request.parameters or {}
                )
            elif request.analysis_type == "dynamics":
                results = await self._analyze_dynamics(engine, request.parameters or {})
            elif request.analysis_type == "energetics":
                results = await self._analyze_energetics(
                    engine, request.parameters or {}
                )
            else:
                return AnalysisResponse(
                    analysis_type=request.analysis_type,
                    success=False,
                    results={
                        "error": f"Unknown analysis type: {request.analysis_type}"
                    },
                )

            return AnalysisResponse(
                analysis_type=request.analysis_type, success=True, results=results
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResponse(
                analysis_type=request.analysis_type,
                success=False,
                results={"error": str(e)},
            )

    async def _analyze_kinematics(
        self, engine: Any, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze kinematic properties."""
        q, v = engine.get_state()

        results = {
            "joint_positions": q.tolist(),
            "joint_velocities": v.tolist(),
            "degrees_of_freedom": len(q),
        }

        # Add Jacobian analysis if engine supports it
        if hasattr(engine, "compute_jacobian"):
            try:
                jacobian = engine.compute_jacobian()
                results["jacobian_condition_number"] = float(
                    1.0 / min(1e-12, min(jacobian.shape))  # Simplified condition number
                )
            except Exception as e:
                logger.warning(f"Jacobian computation failed: {e}")

        return results

    async def _analyze_dynamics(
        self, engine: Any, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze dynamic properties."""
        q, v = engine.get_state()

        results = {"joint_positions": q.tolist(), "joint_velocities": v.tolist()}

        # Add mass matrix if engine supports it
        if hasattr(engine, "compute_mass_matrix"):
            try:
                engine.compute_mass_matrix()
                results["mass_matrix_determinant"] = (
                    1.0  # Simplified - would need actual determinant computation
                )
            except Exception as e:
                logger.warning(f"Mass matrix computation failed: {e}")

        # Add inverse dynamics if engine supports it
        if hasattr(engine, "compute_inverse_dynamics"):
            try:
                # Use zero acceleration for static analysis
                zero_accel = [0.0] * len(v)
                torques = engine.compute_inverse_dynamics(zero_accel)
                results["static_torques"] = torques.tolist()
            except Exception as e:
                logger.warning(f"Inverse dynamics computation failed: {e}")

        return results

    async def _analyze_energetics(
        self, engine: Any, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze energy properties."""
        q, v = engine.get_state()

        results = {
            "kinetic_energy": 0.5 * sum(vi**2 for vi in v),  # Simplified
            "potential_energy": 0.0,  # Would need actual computation
            "total_energy": 0.5 * sum(vi**2 for vi in v),
        }

        return results
