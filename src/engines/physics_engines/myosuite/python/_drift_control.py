from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.core.contracts import check_finite, postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class DriftControlMixin:
    # Attributes provided by EngineInitMixin.__init__; declared here for type checking.
    if TYPE_CHECKING:
        sim: Any

        @property
        def is_initialized(self) -> bool: ...
        def get_state(self) -> tuple[np.ndarray, np.ndarray]: ...
        def set_state(self, q: np.ndarray, v: np.ndarray) -> None: ...
        def compute_mass_matrix(self) -> np.ndarray: ...

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Drift acceleration must contain finite values")
    def compute_drift_acceleration(self) -> np.ndarray:
        if not self.sim:
            logger.warning("Simulation not initialized")
            return np.array([])

        ctrl_saved = self.sim.data.ctrl.copy()
        try:
            self.sim.data.ctrl[:] = 0.0
            self.sim.forward()
            a_drift: np.ndarray = np.array(self.sim.data.qacc)
        finally:
            self.sim.data.ctrl[:] = ctrl_saved
            self.sim.forward()
        return a_drift

    @precondition(lambda self, tau: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Control acceleration must contain finite values")
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        if tau is None:
            raise ValueError("tau must be provided")
        if not self.sim:
            logger.warning("Simulation not initialized")
            return np.array([])

        M = self.compute_mass_matrix()
        if M.size == 0:
            return np.array([])
        a_control = np.linalg.solve(M, tau)
        return a_control

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        if q is None:
            raise ValueError("q must be provided")
        if not self.sim:
            return np.array([])

        q_saved, v_saved = self.get_state()
        ctrl_saved = self.sim.data.ctrl.copy()
        try:
            self.set_state(q, v)
            self.sim.data.ctrl[:] = 0.0
            self.sim.forward()
            a_ztcf = np.array(self.sim.data.qacc)
        finally:
            self.sim.data.ctrl[:] = ctrl_saved
            self.set_state(q_saved, v_saved)
        return a_ztcf

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        if q is None:
            raise ValueError("q must be provided")
        if not self.sim:
            return np.array([])

        q_saved, v_saved = self.get_state()

        try:
            n_v = len(v_saved) if hasattr(v_saved, "__len__") else 1
        except TypeError:
            n_v = 1

        try:
            self.set_state(q, np.zeros(n_v))
            self.sim.forward()
            a_zvcf = np.array(self.sim.data.qacc)
        finally:
            self.set_state(q_saved, v_saved)
        return a_zvcf

    def get_acceleration(self) -> np.ndarray:
        if not self.sim:
            return np.array([])
        return np.array(self.sim.data.qacc)
