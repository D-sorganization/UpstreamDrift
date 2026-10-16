"""MyoSuite closed-loop force adapter: the MuJoCo adapter on a MyoSuite model.

NSX-52 (#12580).  MyoSuite drives the same MJCF through its own ``MujocoEnv``
layer, so the snapshot, drift, topology and parity code is the MuJoCo
adapter's (:class:`MujocoForceAdapter`) and is not re-implemented here.  Only
the model source, the engine identity and the capability gating differ.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from src.engines.physics_engines.myosuite.python.full_body_parity import (
    MyoSuiteFullBodyModel,
)
from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.loop_nullspace.derivative_capability import (
    DerivativeCapabilityUnavailable,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    EngineType,
    MujocoForceAdapter,
)

__all__ = ["MyoSuiteForceAdapter"]

Array = NDArray[np.float64]
_REGULARISATION_GATED = (
    Capability.CLOSED_LOOP_SNAPSHOT,
    Capability.CLOSED_LOOP_DERIVATIVES,
)


class MyoSuiteForceAdapter(MujocoForceAdapter):
    """Spec full-body force adapter over the MyoSuite-held model.

    ``CLOSED_LOOP_SNAPSHOT`` is ``FULL`` only while ``kkt_regularization`` is
    exactly 0 (an exact weld solve, as MuJoCo/Pinocchio/Drake); a Tikhonov
    term makes the engine a different system, so the capability is ``NONE``.
    The value is recorded in the snapshot's model provenance.
    """

    def _make_model(self, spec_bytes: bytes) -> MyoSuiteFullBodyModel:
        return MyoSuiteFullBodyModel(spec_bytes)

    @property
    def engine_type(self) -> EngineType:
        return EngineType.MYOSUITE

    @property
    def kkt_regularization(self) -> float:
        """Tikhonov term of the underlying weld KKT solve (0 means exact)."""
        return float(self._model.kkt_regularization)

    @kkt_regularization.setter
    def kkt_regularization(self, value: float) -> None:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(
                f"kkt_regularization must be finite and >= 0, got {value!r}"
            )
        self._model.kkt_regularization = number

    def capability(self, capability: Capability) -> CapabilityLevel:
        """Snapshot ``FULL`` / derivatives ``PARTIAL`` iff ``kkt_regularization == 0``.

        Otherwise ``NONE``: a Tikhonov term makes the engine a different system.
        """
        if capability in _REGULARISATION_GATED and self.kkt_regularization != 0.0:
            return CapabilityLevel.NONE
        return super().capability(capability)

    def constrained_acceleration_derivatives(
        self, q: Array, v: Array, tau: Array, *, step: float = 1e-5
    ) -> Any:
        """Hard-KKT finite-difference derivatives; refuses unless regularisation is 0.

        Raises :class:`DerivativeCapabilityUnavailable` when
        ``kkt_regularization != 0`` (the derivatives would be of a regularised
        solve, not of the rigid closure).  Muscle activation states are
        unsupported until NSX-94.  Otherwise identical to the MuJoCo adapter.
        """
        if self.kkt_regularization != 0.0:
            raise DerivativeCapabilityUnavailable(
                "MyoSuite closed-loop derivatives require kkt_regularization == 0 "
                f"(exact rigid weld solve); got {self.kkt_regularization:g}"
            )
        return super().constrained_acceleration_derivatives(q, v, tau, step=step)
