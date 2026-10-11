"""Per-frame static optimisation of the leg muscles against reference efforts.

Part of issue #11617 (epic #11605), phase 2.  The reference efforts are the joint
torques that produced the (forward-dynamics-consistent) reference motion, so the
required leg joint moments are known exactly and no ground-force estimate is
needed.  At every sampled frame the muscle redundancy is resolved with the same
cost OpenSim's ``StaticOptimization`` uses (sum of squared activations, with
unit-optimal-force reserve actuators on the leg coordinates)::

    min_a  sum a_m^2 + w * sum r_c^2     0 <= a_m <= 1
    r_c = tau_c - sum_m R_mc F_m(a_m),   F_m(a) = F_pas,m + a * F_act,m

``F_pas``/``F_act`` come from the rigid-tendon Millard muscle at the sampled
state and ``R`` are the muscle moment arms.  The AnalyzeTool route is not used
because it differences accelerations itself and cannot represent the grip
loop-closure reaction that acts on the arm coordinates of the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from src.engines.physics_engines.opensim.python import musculoskeletal_graft as graft
from src.shared.python.contracts import require

LEG_JOINTS: tuple[str, ...] = (
    "hip_flexion",
    "hip_adduction",
    "hip_rotation",
    "knee_angle",
    "ankle_angle",
    "subtalar_angle",
    "mtp_angle",
)


def leg_coordinates(side: str | None = None) -> tuple[str, ...]:
    """Spec leg coordinate names (both sides, or one side)."""
    sides = graft.SIDES if side is None else (side,)
    return tuple(f"{j}_{s}" for s in sides for j in LEG_JOINTS)


@dataclass(frozen=True)
class FrameSolution:
    """Result of one frame: activations, reserves and the optimiser status."""

    activation: np.ndarray
    reserve: np.ndarray
    cost: float
    success: bool


def solve_frame(
    active: np.ndarray,
    passive: np.ndarray,
    moment: np.ndarray,
    tau: np.ndarray,
    *,
    reserve_weight: float = 1.0,
) -> FrameSolution:
    """Resolve muscle redundancy for one frame.

    Args:
        active: ``(nm,)`` force per unit activation (N), >= 0.
        passive: ``(nm,)`` force at zero activation (N), >= 0.
        moment: ``(nm, nc)`` moment arms (m); ``moment[m, c] * F_m`` is the
            generalised force of muscle ``m`` on coordinate ``c``.
        tau: ``(nc,)`` required generalised forces (N m).
        reserve_weight: relative weight of the squared reserve in the cost.

    Returns:
        The minimiser.  Postconditions: ``0 <= activation <= 1`` and
        ``reserve = tau - moment.T @ (passive + activation * active)``.

    Raises:
        ValueError: on shape mismatch, non-finite input or a non-positive weight.
    """
    nm = active.shape[0]
    require(passive.shape == (nm,), "passive must be (nm,)")
    require(moment.ndim == 2 and moment.shape[0] == nm, "moment must be (nm, nc)")
    require(tau.shape == (moment.shape[1],), "tau must be (nc,)")
    require(reserve_weight > 0.0, "reserve_weight must be positive")
    for name, array in (
        ("active", active),
        ("passive", passive),
        ("moment", moment),
        ("tau", tau),
    ):
        require(bool(np.isfinite(array).all()), f"{name} must be finite")
    gain = moment.T * active[None, :]  # (nc, nm)
    offset = moment.T @ passive
    root = np.sqrt(reserve_weight)
    design = np.vstack([np.eye(nm), root * gain])
    target = np.concatenate([np.zeros(nm), root * (tau - offset)])
    result = lsq_linear(design, target, bounds=(0.0, 1.0), method="bvls")
    a = np.clip(result.x, 0.0, 1.0)
    reserve = tau - offset - gain @ a
    cost = float(a @ a + reserve_weight * (reserve @ reserve))
    return FrameSolution(a, reserve, cost, bool(result.success))


class MuscleBasis:
    """Muscle forces and moment arms of a grafted model at prescribed states.

    Args:
        model: model from ``build_spec_musculoskeletal_model``.
        coordinate_order: the 44 spec coordinate names of the bundle.

    ``evaluate(q, v)`` returns ``(active, passive, moment)`` for the 80 muscles
    and the 14 leg coordinates (left then right ordering of ``leg_coordinates``).
    """

    def __init__(self, model: Any, coordinate_order: tuple[str, ...]) -> None:
        self.model = model
        self.state = model.initSystem()
        self.coordinate_order = coordinate_order
        self.muscles = list(model.getMuscles())
        self.muscle_names = [m.getName() for m in self.muscles]
        self.coords = leg_coordinates()
        coordinate_set = model.getCoordinateSet()
        self._coordinate = {n: coordinate_set.get(n) for n in coordinate_order}
        self._beta = {
            s: coordinate_set.get(f"knee_angle_{s}_beta") for s in graft.SIDES
        }
        self._leg_objects = [coordinate_set.get(n) for n in self.coords]
        self._crossing = self._side_masks()

    def _side_masks(self) -> np.ndarray:
        """Boolean ``(nm, nc)``: muscle ``m`` may act on coordinate ``c`` (same side)."""
        mask = np.zeros((len(self.muscles), len(self.coords)), dtype=bool)
        for i, name in enumerate(self.muscle_names):
            for j, coord in enumerate(self.coords):
                mask[i, j] = graft.side_of(name) == graft.side_of(coord)
        return mask

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Pose the model (spec order); the patella follows the knee 1:1."""
        require(q.shape == (len(self.coordinate_order),), "q has the wrong size")
        for i, name in enumerate(self.coordinate_order):
            coordinate = self._coordinate[name]
            coordinate.setValue(self.state, float(q[i]), False)
            coordinate.setSpeedValue(self.state, float(v[i]))
        for side, beta in self._beta.items():
            i = self.coordinate_order.index(f"knee_angle_{side}")
            beta.setValue(self.state, float(q[i]), False)
            beta.setSpeedValue(self.state, float(v[i]))

    def evaluate(
        self, q: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(active (nm,), passive (nm,), moment (nm, nc))`` at ``(q, v)``."""
        self.set_state(q, v)
        for muscle in self.muscles:
            muscle.setActivation(self.state, 0.0)
        self.model.realizeDynamics(self.state)
        passive = np.array([m.getActuation(self.state) for m in self.muscles])
        for muscle in self.muscles:
            muscle.setActivation(self.state, 1.0)
        self.model.realizeDynamics(self.state)
        full = np.array([m.getActuation(self.state) for m in self.muscles])
        moment = np.zeros((len(self.muscles), len(self.coords)))
        for i, muscle in enumerate(self.muscles):
            for j in np.flatnonzero(self._crossing[i]):
                moment[i, j] = muscle.computeMomentArm(self.state, self._leg_objects[j])
        return np.maximum(full - passive, 0.0), np.maximum(passive, 0.0), moment


def optimal_forces(model: Any) -> np.ndarray:
    """Maximum isometric force of every muscle (N), in model order."""
    return np.array([m.getMaxIsometricForce() for m in model.getMuscles()])
