from __future__ import annotations

import numpy as np

from src.shared.python.core.contracts import check_finite, postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class DriftControlMixin:
    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Drift acceleration must contain finite values")
    def compute_drift_acceleration(self) -> np.ndarray:
        if not self.sim:
            logger.warning("Simulation not initialized")
            return np.array([])

        try:
            ctrl_saved = self.sim.data.ctrl.copy()
            self.sim.data.ctrl[:] = 0.0
            self.sim.forward()
            a_drift: np.ndarray = np.array(self.sim.data.qacc)
            self.sim.data.ctrl[:] = ctrl_saved
            self.sim.forward()
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
        if not self.sim:
            logger.warning("Simulation not initialized")
            return np.array([])

        try:
            M = self.compute_mass_matrix()
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
        if not self.sim:
            return np.array([])

        try:
            q_saved, v_saved = self.get_state()
            ctrl_saved = self.sim.data.ctrl.copy()
            self.set_state(q, v)
            self.sim.data.ctrl[:] = 0.0
            self.sim.forward()
            a_ztcf = np.array(self.sim.data.qacc)
            self.sim.data.ctrl[:] = ctrl_saved
            self.set_state(q_saved, v_saved)
            return a_ztcf

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute ZTCF: {e}")
            return np.array([])

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        if not (q is not None):
            raise ValueError("q must be provided")
        if not (q is not None):
            raise ValueError("q must be provided")
        if not self.sim:
            return np.array([])

        try:
            q_saved, v_saved = self.get_state()

            try:
                n_v = len(v_saved) if hasattr(v_saved, "__len__") else 1
            except TypeError:
                n_v = 1

            self.set_state(q, np.zeros(n_v))
            self.sim.forward()
            a_zvcf = np.array(self.sim.data.qacc)
            self.set_state(q_saved, v_saved)
            return a_zvcf

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute ZVCF: {e}")
            return np.array([])

    def get_acceleration(self) -> np.ndarray:
        if not self.sim:
            return np.array([])
        return np.array(self.sim.data.qacc)
