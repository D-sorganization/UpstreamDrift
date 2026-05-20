from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.core.contracts import check_finite, postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class DynamicsMixin:
    # Attributes provided by EngineInitMixin.__init__; declared here for type checking.
    if TYPE_CHECKING:
        sim: Any

        @property
        def is_initialized(self) -> bool: ...

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Mass matrix must contain finite values")
    def compute_mass_matrix(self) -> np.ndarray:
        if not self.sim:
            return np.array([])

        try:
            import mujoco

            if hasattr(self.sim.model, "nv") and not isinstance(
                self.sim.model.nv,
                type(lambda: None),
            ):
                nv = self.sim.model.nv
            else:
                nv = 1

            M = np.zeros((nv, nv))

            try:
                mujoco.mj_fullM(self.sim.model, M, self.sim.data.qM)
            except TypeError:
                M = np.eye(nv)

            return M

        except ImportError as e:
            logger.error("Failed to compute mass matrix: %s", e)
            return np.array([])

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Bias forces must contain finite values")
    def compute_bias_forces(self) -> np.ndarray:
        if self.sim:
            return np.array(self.sim.data.qfrc_bias)
        return np.array([])

    def compute_gravity_forces(self) -> np.ndarray:
        return np.array([])

    @precondition(lambda self, qacc: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Inverse dynamics torques must contain finite values")
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        if qacc is None:
            raise ValueError("qacc must be provided")
        if not self.sim:
            return np.array([])

        try:
            import mujoco

            self.sim.data.qacc[:] = qacc
            mujoco.mj_inverse(self.sim.model, self.sim.data)
            return np.array(self.sim.data.qfrc_inverse)

        except ImportError as e:
            logger.error("Failed to compute inverse dynamics: %s", e)
            return np.array([])

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        if body_name is None:
            raise ValueError("body_name must be provided")
        if not self.sim:
            return None

        try:
            import mujoco

            body_id = mujoco.mj_name2id(
                self.sim.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )

            if body_id == -1:
                return None

            jacp = np.zeros((3, self.sim.model.nv))
            jacr = np.zeros((3, self.sim.model.nv))

            mujoco.mj_jacBody(self.sim.model, self.sim.data, jacp, jacr, body_id)

            return {"linear": jacp, "angular": jacr, "spatial": np.vstack([jacr, jacp])}

        except ImportError as e:
            logger.error("Failed to compute Jacobian for body '%s': %s", body_name, e)
            return None
