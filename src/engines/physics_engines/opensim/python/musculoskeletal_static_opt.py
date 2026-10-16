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

The model-independent numerics (``FrameSolution``, ``solve_frame``) live in
``static_opt_core`` (SPN-10b, #12761), shared with the trunk static
optimisation; they are re-exported here unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import musculoskeletal_graft as graft
from src.engines.physics_engines.opensim.python.static_opt_core import (
    FrameSolution,
    solve_frame,
)
from src.shared.python.contracts import require

__all__ = [
    "LEG_JOINTS",
    "FrameSolution",
    "MuscleBasis",
    "leg_coordinates",
    "optimal_forces",
    "solve_frame",
]

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
