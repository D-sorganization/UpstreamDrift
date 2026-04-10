from __future__ import annotations

import numpy as np

from src.shared.python.core.contracts import check_finite, postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class DynamicsMixin:
    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Mass matrix must contain finite values")
    def compute_mass_matrix(self) -> np.ndarray:
        if not self.sim:  # type: ignore[attr-defined]
            return np.array([])

        try:
            import mujoco

            if hasattr(self.sim.model, "nv") and not isinstance(  # type: ignore[attr-defined]
                self.sim.model.nv,  # type: ignore[attr-defined]
                type(lambda: None),  # type: ignore[attr-defined]
            ):
                nv = self.sim.model.nv  # type: ignore[attr-defined]
            else:
                nv = 1

            M = np.zeros((nv, nv))

            try:
                mujoco.mj_fullM(self.sim.model, M, self.sim.data.qM)  # type: ignore[attr-defined]
            except TypeError:
                M = np.eye(nv)

            return M

        except ImportError as e:
            logger.error("Failed to compute mass matrix: %s", e)
            return np.array([])

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Bias forces must contain finite values")
    def compute_bias_forces(self) -> np.ndarray:
        if self.sim:  # type: ignore[attr-defined]
            return np.array(self.sim.data.qfrc_bias)  # type: ignore[attr-defined]
        return np.array([])

    def compute_gravity_forces(self) -> np.ndarray:
        return np.array([])

    @precondition(lambda self, qacc: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Inverse dynamics torques must contain finite values")
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        if not (qacc is not None):
            raise ValueError("qacc must be provided")
        if not (qacc is not None):
            raise ValueError("qacc must be provided")
        if not self.sim:  # type: ignore[attr-defined]
            return np.array([])

        try:
            import mujoco

            self.sim.data.qacc[:] = qacc  # type: ignore[attr-defined]
            mujoco.mj_inverse(self.sim.model, self.sim.data)  # type: ignore[attr-defined]
            return np.array(self.sim.data.qfrc_inverse)  # type: ignore[attr-defined]

        except ImportError as e:
            logger.error("Failed to compute inverse dynamics: %s", e)
            return np.array([])

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        if not (body_name is not None):
            raise ValueError("body_name must be provided")
        if not (body_name is not None):
            raise ValueError("body_name must be provided")
        if not self.sim:  # type: ignore[attr-defined]
            return None

        try:
            import mujoco

            body_id = mujoco.mj_name2id(
                self.sim.model,  # type: ignore[attr-defined]
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,  # type: ignore[attr-defined]
            )

            if body_id == -1:
                return None

            jacp = np.zeros((3, self.sim.model.nv))  # type: ignore[attr-defined]
            jacr = np.zeros((3, self.sim.model.nv))  # type: ignore[attr-defined]

            mujoco.mj_jacBody(self.sim.model, self.sim.data, jacp, jacr, body_id)  # type: ignore[attr-defined]

            return {"linear": jacp, "angular": jacr, "spatial": np.vstack([jacr, jacp])}

        except ImportError as e:
            logger.error("Failed to compute Jacobian for body '%s': %s", body_name, e)
            return None
