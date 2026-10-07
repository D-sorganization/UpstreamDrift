"""Vector view of the spec-built full-body adapters for same-input parity.

Each engine adapter takes and returns name-keyed dictionaries in its own
internal order.  :class:`VectorPlant` fixes one convention for every engine,
the spec ``coordinate_order``, and exposes only what the parity integrator
needs: constrained accelerations and the dual-grip closure residuals.  It uses
the adapters' public APIs, so the dynamics are each engine's own (#11606).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

PARITY_ENGINES: tuple[str, ...] = ("mujoco", "drake", "pinocchio")


def _build_adapter(engine: str, spec_bytes: bytes, kkt_regularization: float) -> Any:
    spec = json.loads(spec_bytes)
    if engine == "mujoco":
        from src.engines.physics_engines.mujoco.python.full_body_model import (
            NativeMujocoFullBodyModel,
        )

        adapter = NativeMujocoFullBodyModel(spec_bytes)
        adapter.kkt_regularization = float(kkt_regularization)
        return adapter
    if engine == "drake":
        from src.engines.physics_engines.drake.python.full_body_model import (
            FullBodyDrakeModel,
        )

        return FullBodyDrakeModel(spec)
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    return FullBodyPinocchioModel(spec)


class VectorPlant:
    """One engine's full-body dynamics in spec coordinate order.

    Args:
        engine: One of :data:`PARITY_ENGINES`.
        spec_bytes: Full-body spec document (the same bytes for every engine).
        kkt_regularization: MuJoCo weld-KKT Tikhonov term; 0 gives the exact
            solve used by Drake and Pinocchio.  Ignored by the other engines.

    Postcondition: ``coordinate_order`` equals the spec ``coordinate_order``.
    """

    def __init__(
        self, engine: str, spec_bytes: bytes, *, kkt_regularization: float = 0.0
    ) -> None:
        if engine not in PARITY_ENGINES:
            raise ValueError(f"engine must be one of {PARITY_ENGINES}, got {engine!r}")
        if not isinstance(spec_bytes, bytes | bytearray):
            raise TypeError("spec_bytes must be the raw spec document bytes")
        if not np.isfinite(kkt_regularization) or kkt_regularization < 0.0:
            raise ValueError("kkt_regularization must be finite and non-negative")
        self.engine = engine
        self.kkt_regularization = float(kkt_regularization)
        self.coordinate_order: tuple[str, ...] = tuple(
            json.loads(spec_bytes)["coordinate_order"]
        )
        self.nv = len(self.coordinate_order)
        self._adapter = _build_adapter(engine, bytes(spec_bytes), kkt_regularization)
        if tuple(self._adapter.coordinate_order) != self.coordinate_order:
            raise ValueError(f"{engine} adapter coordinate order differs from spec")
        self._zeros = np.zeros(self.nv)

    def _named(self, values: Array) -> dict[str, float]:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (self.nv,) or not np.isfinite(vector).all():
            raise ValueError(f"Expected {self.nv} finite values in spec order")
        return dict(zip(self.coordinate_order, map(float, vector), strict=True))

    def _ordered(self, values: Mapping[str, float]) -> Array:
        return np.array([values[name] for name in self.coordinate_order], dtype=float)

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array:
        """Constrained generalized acceleration for efforts ``tau`` (spec order)."""
        result = self._adapter.accelerations(
            self._named(q), self._named(v), self._named(tau)
        )
        return self._ordered(result)

    def closure_residuals(self, q: Array, v: Array) -> tuple[Array, Array]:
        """Engine-native weld pose residual (6) and rate residual (6) at ``(q, v)``.

        Each engine uses its own frame convention.  Only the zero set of the
        pose residual and the row space of the rate map are convention-free,
        which is all :func:`project_to_closure` relies on.
        """
        if self.engine == "mujoco":
            self._adapter.accelerations(
                self._named(q), self._named(v), self._named(self._zeros)
            )
            pose, rate = self._adapter.closure_errors()
        else:
            pose, rate = self._adapter.closure_residuals(self._named(q), self._named(v))
        return np.asarray(pose, dtype=float), np.asarray(rate, dtype=float)

    def closure_pose_residual(self, q: Array) -> Array:
        """Weld pose residual at ``q``; zero exactly on the closure manifold."""
        return self.closure_residuals(q, self._zeros)[0]

    def closure_rate_matrix(self, q: Array) -> Array:
        """Matrix ``D`` (6 x nv) with rate residual ``D @ v`` at ``q``.

        The rate residual is linear in ``v``, so its columns are recovered
        exactly from unit velocities.
        """
        columns = []
        for k in range(self.nv):
            unit = np.zeros(self.nv)
            unit[k] = 1.0
            columns.append(self.closure_residuals(q, unit)[1])
        return np.column_stack(columns)

    def kinematic_frames(self, q: Array) -> dict[str, Array]:
        """Engine forward kinematics of the spec frames at ``q`` (4x4 poses)."""
        return {
            name: np.asarray(pose, dtype=float)
            for name, pose in self._adapter.frame_poses(self._named(q)).items()
        }
