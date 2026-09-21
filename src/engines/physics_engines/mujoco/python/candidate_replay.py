"""MuJoCo same-input replay using shared contact and rigid grip dynamics."""

from __future__ import annotations

import json
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.engines.physics_engines.mujoco.python.replay_contract import ReplaySettings
from src.shared.python.contracts import precondition
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.polynomial_actuation import ROOT_COORDINATES


class AccelerationPlant(Protocol):
    def acceleration(
        self, q: np.ndarray, v: np.ndarray, effort: np.ndarray
    ) -> np.ndarray: ...


class ReplayPlant:
    """Name-safe MuJoCo plant; engine stock contact remains disabled."""

    @precondition(
        lambda document: isinstance(document, dict), "Model document required"
    )
    def __init__(
        self, document: dict, ground_height_m: float, settings: ReplaySettings
    ) -> None:
        self.adapter = NativeMujocoFullBodyModel(json.dumps(document).encode())
        self.names = tuple(self.adapter.coordinate_order)
        self.actuated = np.array([name not in ROOT_COORDINATES for name in self.names])
        model = self.adapter.model
        self.indices = np.array(
            [int(model.joint(name).dofadr[0]) for name in self.names]
        )
        model.dof_armature[self.indices] = self.actuated * settings.armature_kg_m2
        ground = self.adapter.ground_plane
        self.adapter.ground_plane = GroundPlane(ground.normal, ground_height_m)

    @precondition(lambda q: np.isfinite(q).all(), "Finite state required")
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Mass matrix in candidate coordinate order, including declared armature."""
        import mujoco

        model, data = self.adapter.model, self.adapter.data
        self.adapter.generalized_forces(
            dict(zip(self.names, q, strict=True)), dict.fromkeys(self.names, 0.0)
        )
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        return mass[np.ix_(self.indices, self.indices)]

    @precondition(
        lambda q, v, effort: all(np.isfinite(x).all() for x in (q, v, effort)),
        "Finite dynamics inputs required",
    )
    def acceleration(
        self, q: np.ndarray, v: np.ndarray, effort: np.ndarray
    ) -> np.ndarray:
        """Constrained acceleration with ground forces recomputed at the live state."""
        result = self.adapter.accelerations(
            *[dict(zip(self.names, x, strict=True)) for x in (q, v, effort)]
        )
        return np.array([result[name] for name in self.names])


@precondition(lambda times: len(times) >= 2, "Replay needs at least two frames")
def replay_controls(
    plant: AccelerationPlant,
    times: np.ndarray,
    q0: np.ndarray,
    v0: np.ndarray,
    efforts: np.ndarray,
    settings: ReplaySettings,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Integrate held controls from q0/v0; return only completed frames on failure.

    Interval restarts change control, never reset the state to reference poses.
    The final saved control is unused because it has no following interval.
    """
    n = len(q0)
    if v0.shape != q0.shape or efforts.shape != (len(times), n):
        raise ValueError("Replay state/control shape mismatch")
    if not all(np.isfinite(x).all() for x in (times, q0, v0, efforts)) or np.any(
        np.diff(times) <= 0
    ):
        raise ValueError("Finite inputs and strictly increasing time required")
    states = [np.concatenate([q0, v0])]
    evaluations = 0
    for k in range(len(times) - 1):

        def rhs(t: float, state: np.ndarray, interval: int = k) -> np.ndarray:
            nonlocal evaluations
            evaluations += 1
            if evaluations > settings.max_evaluations:
                raise RuntimeError("Replay evaluation budget exhausted")
            return np.concatenate(
                [state[n:], plant.acceleration(state[:n], state[n:], efforts[interval])]
            )

        try:
            solution = solve_ivp(
                rhs,
                (times[k], times[k + 1]),
                states[-1],
                method="DOP853",
                rtol=settings.rtol,
                atol=settings.atol,
                max_step=settings.max_step_s,
                t_eval=[times[k + 1]],
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            state = solution.y[:, -1]
            if not np.isfinite(state).all():
                raise FloatingPointError("Nonfinite replay state")
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            result = np.asarray(states)
            return result[:, :n], result[:, n:], str(exc)
        states.append(state)
    result = np.asarray(states)
    return result[:, :n], result[:, n:], None
