from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class SimulationCoreMixin:
    # Attributes provided by EngineInitMixin.__init__; declared here for type checking.
    if TYPE_CHECKING:
        env: Any
        sim: Any
        _terminated: bool
        _dt: float

        @property
        def is_initialized(self) -> bool: ...

        _last_action: np.ndarray | None

    @precondition(lambda self: self.env is not None, "Environment must be loaded")
    def reset(self) -> None:
        if self.env:
            self.env.reset()
            self._terminated = False

    @precondition(
        lambda self, dt=None: self.env is not None, "Environment must be loaded"
    )
    def step(self, dt: float | None = None) -> None:
        if not self.env:
            return

        if dt is not None and dt != getattr(self, "_dt", None):
            logger.warning(
                "Runtime timestep modification is unsafe in MuJoCo. Ignoring dt override."
            )

        if self._terminated:
            logger.warning(
                "step() called on a terminated MyoSuite environment; "
                "call reset() before stepping again"
            )
            return

        action = getattr(self, "_last_action", None)
        if action is None:
            action = self.env.action_space.sample() * 0.0

        result = self.env.step(action)
        if len(result) >= 4:
            _obs, _reward, terminated, truncated = result[:4]
            self._terminated = bool(terminated or truncated)

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    def forward(self) -> None:
        if self.sim:
            self.sim.forward()

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.sim:
            return np.array([]), np.array([])
        return (np.array(self.sim.data.qpos[:]), np.array(self.sim.data.qvel[:]))

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        if q is None:
            raise ValueError("q must be provided")
        if not self.sim:
            return

        q = np.atleast_1d(q)
        v = np.atleast_1d(v)

        try:
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel

            if hasattr(qpos, "__len__") and hasattr(qvel, "__len__"):
                if len(q) == len(qpos):
                    self.sim.data.qpos[:] = q
                if len(v) == len(qvel):
                    self.sim.data.qvel[:] = v
            else:
                self.sim.data.qpos[:] = q
                self.sim.data.qvel[:] = v

        except (TypeError, AttributeError) as e:
            logger.debug(f"Primary state assignment failed (may be mocked): {e}")

            try:
                self.sim.data.qpos[:] = q
                self.sim.data.qvel[:] = v
            except (RuntimeError, ValueError, OSError) as fallback_error:
                logger.debug(f"Fallback state assignment failed: {fallback_error}")

        try:
            self.sim.forward()
        except (RuntimeError, ValueError, OSError) as forward_error:
            logger.debug(f"Forward dynamics failed (may be mocked): {forward_error}")

    def set_control(self, u: np.ndarray) -> None:
        if u is None:
            raise ValueError("u must be provided")
        self._last_action = np.array(u, copy=True)

        if not self.sim:
            return

        try:
            ctrl = self.sim.data.ctrl

            if hasattr(ctrl, "shape") and hasattr(ctrl, "__len__"):
                if len(u) == ctrl.shape[0]:
                    self.sim.data.ctrl[:] = u
            else:
                self.sim.data.ctrl[:] = u

        except (TypeError, AttributeError) as e:
            logger.debug(f"Primary control assignment failed (may be mocked): {e}")

            try:
                self.sim.data.ctrl[:] = u
            except (RuntimeError, ValueError, OSError) as fallback_error:
                logger.debug(f"Fallback control assignment failed: {fallback_error}")

    def get_time(self) -> float:
        if self.sim:
            return float(self.sim.data.time)
        return 0.0
