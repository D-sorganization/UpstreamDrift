from __future__ import annotations

import numpy as np

from src.shared.python.core.contracts import check_finite, postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class DriftControlMixin:
    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Drift acceleration must contain finite values")
    def compute_drift_acceleration(self) -> np.ndarray:
        if not self.sim:  # type: ignore[attr-defined]
            logger.warning("Simulation not initialized")
            return np.array([])

        try:
            ctrl_saved = self.sim.data.ctrl.copy()  # type: ignore[attr-defined]
            self.sim.data.ctrl[:] = 0.0  # type: ignore[attr-defined]
            self.sim.forward()  # type: ignore[attr-defined]
            a_drift: np.ndarray = np.array(self.sim.data.qacc)  # type: ignore[attr-defined]
            self.sim.data.ctrl[:] = ctrl_saved  # type: ignore[attr-defined]
            self.sim.forward()  # type: ignore[attr-defined]
            return a_drift

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute drift acceleration: {e}")
            return np.array([])

    @precondition(lambda self, tau: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Control acceleration must contain finite values")
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        if not (tau is not None):
            raise ValueError("tau must be provided")
        if not (tau is not None):
            raise ValueError("tau must be provided")
        if not self.sim:  # type: ignore[attr-defined]
            logger.warning("Simulation not initialized")
            return np.array([])

        try:
            M = self.compute_mass_matrix()  # type: ignore[attr-defined]
            if M.size == 0:
                return np.array([])
            a_control = np.linalg.solve(M, tau)
            return a_control

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute control acceleration: {e}")
            return np.zeros_like(tau)

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        if not (q is not None):
            raise ValueError("q must be provided")
        if not (q is not None):
            raise ValueError("q must be provided")
        if not self.sim:  # type: ignore[attr-defined]
            return np.array([])

        try:
            q_saved, v_saved = self.get_state()  # type: ignore[attr-defined]
            ctrl_saved = self.sim.data.ctrl.copy()  # type: ignore[attr-defined]
            self.set_state(q, v)  # type: ignore[attr-defined]
            self.sim.data.ctrl[:] = 0.0  # type: ignore[attr-defined]
            self.sim.forward()  # type: ignore[attr-defined]
            a_ztcf = np.array(self.sim.data.qacc)  # type: ignore[attr-defined]
            self.sim.data.ctrl[:] = ctrl_saved  # type: ignore[attr-defined]
            self.set_state(q_saved, v_saved)  # type: ignore[attr-defined]
            return a_ztcf

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute ZTCF: {e}")
            return np.array([])

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        if not (q is not None):
            raise ValueError("q must be provided")
        if not (q is not None):
            raise ValueError("q must be provided")
        if not self.sim:  # type: ignore[attr-defined]
            return np.array([])

        try:
            q_saved, v_saved = self.get_state()  # type: ignore[attr-defined]

            try:
                n_v = len(v_saved) if hasattr(v_saved, "__len__") else 1
            except TypeError:
                n_v = 1

            self.set_state(q, np.zeros(n_v))  # type: ignore[attr-defined]
            self.sim.forward()  # type: ignore[attr-defined]
            a_zvcf = np.array(self.sim.data.qacc)  # type: ignore[attr-defined]
            self.set_state(q_saved, v_saved)  # type: ignore[attr-defined]
            return a_zvcf

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute ZVCF: {e}")
            return np.array([])

    def get_acceleration(self) -> np.ndarray:
        if not self.sim:  # type: ignore[attr-defined]
            return np.array([])
        return np.array(self.sim.data.qacc)  # type: ignore[attr-defined]
