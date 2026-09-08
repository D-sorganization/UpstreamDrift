"""``SymbolicSwingModel``: the seven-DOF swing as CasADi functions (Phase 1.1).

One place turns :class:`GolferModel` + :class:`ClubModel` into the CasADi
``Function`` objects bioptim's custom-model protocol needs -- with the
**same anthropometric inertials** the URDF bridge emits (#9755), so the
OCPs built on top of it bound physically meaningful torques.

Every function takes a trailing ``parameters`` input: the vector of the
model's declared symbolic parameters (segment lengths / masses, Phase 4),
empty when none were declared. That matches how bioptim's penalties call a
model (``model.markers()(q, parameters)``) and keeps every ``Function``
free of stray symbols.

Dynamics kernels reuse :func:`casadi_backend.rnea_expression`; the geometry
and inertial formulas are :func:`model_provider.swing_segment_offsets` and
:func:`model_provider.swing_segment_inertials`, evaluated on floats or
symbols alike. Pinocchio validates every kernel numerically in
``ocp/tests/test_symbolic_model.py``; it is never imported here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import ClubModel, GolferModel
from src.shared.python.optimization.casadi_backend import (
    _AXES,
    _GRAVITY,
    require_casadi,
    rnea_expression,
)
from src.shared.python.optimization.model_provider import (
    swing_joint_axes,
    swing_segment_inertials,
    swing_segment_offsets,
)

__all__ = ["MARKER_NAMES", "PARAMETER_NAMES", "SymbolicSwingModel"]

#: Model quantities that may be promoted to symbolic parameters (Phase 4).
PARAMETER_NAMES: tuple[str, ...] = (
    "arm_length",
    "trunk_length",
    "club_length",
    "mass",
    "head_mass",
)

#: Marker set: joint centres (shoulder once -- its two DOFs are co-located),
#: the clubhead at the chain tip, and ``clubface``, offset from the shaft
#: axis. ``keypoint_map`` maps video keypoint names onto these.
#:
#: The ``clubface`` offset is not decoration. Every joint-origin marker is
#: invariant under the terminal shaft-roll DOF (``wrist_rotation`` turns
#: about the axis the clubhead origin sits on), so a marker set made only of
#: joint centres leaves that DOF unobservable and the tracking OCP recovers
#: an arbitrary value for it. A point off the shaft axis -- which is where a
#: real clubhead's mass sits, cf. the MacKenzie 2012 epic -- makes the roll
#: observable.
MARKER_NAMES: tuple[str, ...] = (
    "hip",
    "trunk",
    "shoulder",
    "elbow",
    "wrist",
    "clubhead",
    "clubface",
)
_MARKER_JOINT: dict[str, str] = {
    "hip": "hip_rotation",
    "trunk": "trunk_rotation",
    "shoulder": "shoulder_horizontal",
    "elbow": "elbow_flexion",
    "wrist": "wrist_cock",
    "clubhead": "wrist_rotation",
}
#: Lateral offset [m] of ``clubface`` from the clubhead origin, in the
#: terminal joint frame (perpendicular to the shaft).
CLUBFACE_OFFSET: tuple[float, float, float] = (0.05, 0.0, 0.0)


def _rotation(ca: Any, axis: np.ndarray, angle: Any) -> Any:
    k = ca.SX(axis)
    kx = ca.skew(k)
    return ca.SX.eye(3) + ca.sin(angle) * kx + (1 - ca.cos(angle)) * (kx @ kx)


def _tensor(ca: Any, inertia: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
    ixx, iyy, izz, ixy, ixz, iyz = inertia
    return ca.vertcat(
        ca.horzcat(ixx, ixy, ixz),
        ca.horzcat(ixy, iyy, iyz),
        ca.horzcat(ixz, iyz, izz),
    )


class SymbolicSwingModel:
    """CasADi kinematics and dynamics of the seven-DOF swing chain.

    Args:
        golfer: Anthropometrics; defaults to :class:`GolferModel`.
        club: Club parameters; defaults to :class:`ClubModel`.
        parameters: Names from :data:`PARAMETER_NAMES` to expose as symbolic
            parameters (in this order). Their numeric values are then read
            from the ``parameters`` input of every function instead of the
            golfer/club models; :meth:`nominal_parameters` gives the values
            that reproduce the numeric model.
        ca: The ``casadi`` module (imported on demand when ``None``).

    Postconditions (Pinocchio-validated): ``rnea`` matches ``pin.rnea``,
    ``mass_matrix`` matches ``pin.crba``, ``forward_dynamics`` matches
    ``pin.aba``, ``markers`` match the joint frame placements, and
    ``center_of_mass`` matches ``pin.centerOfMass`` on
    ``model_provider.build_pinocchio_model(golfer, club)``.
    """

    def __init__(
        self,
        golfer: GolferModel | None = None,
        club: ClubModel | None = None,
        *,
        parameters: Sequence[str] = (),
        ca: Any | None = None,
    ) -> None:
        ca = ca or require_casadi()
        self._ca = ca
        self.golfer = golfer or GolferModel()
        self.club = club or ClubModel()
        names = tuple(parameters)
        unknown = [name for name in names if name not in PARAMETER_NAMES]
        if unknown:
            raise ValueError(
                f"unknown parameters {unknown}; allowed: {PARAMETER_NAMES}"
            )
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        self.parameter_names: tuple[str, ...] = names
        self.n_q = len(JOINTS)
        self.dof_names: tuple[str, ...] = tuple(JOINTS)
        self.marker_names: tuple[str, ...] = MARKER_NAMES

        n = self.n_q
        self._q = ca.SX.sym("q", n)
        self._v = ca.SX.sym("v", n)
        self._a = ca.SX.sym("a", n)
        self._tau = ca.SX.sym("tau", n)
        self._p = ca.SX.sym("p", len(names)) if names else ca.SX(0, 1)

        values: dict[str, Any] = {
            "height": self.golfer.height,
            "trunk_length": self.golfer.trunk_length,
            "arm_length": self.golfer.arm_length,
            "club_length": self.club.total_length,
            "mass": self.golfer.mass,
            "head_mass": self.club.head_mass,
        }
        for index, name in enumerate(names):
            values[name] = self._p[index]

        offsets = swing_segment_offsets(
            height=values["height"],
            trunk_length=values["trunk_length"],
            arm_length=values["arm_length"],
            club_length=values["club_length"],
        )
        raw_inertials = swing_segment_inertials(
            offsets=offsets,
            mass=values["mass"],
            trunk_mass_ratio=self.golfer.trunk_mass_ratio,
            arm_mass_ratio=self.golfer.arm_mass_ratio,
            height=values["height"],
            grip_mass=self.club.grip_mass,
            shaft_mass=self.club.shaft_mass,
            shaft_length=self.club.shaft_length,
            head_mass=values["head_mass"],
        )
        axes_by_joint = swing_joint_axes()
        self._offsets = [ca.vertcat(*offsets[name]) for name in JOINTS]
        self._axes = [_AXES[axes_by_joint[name]] for name in JOINTS]
        self._inertials = [
            (mass, ca.vertcat(*com), _tensor(ca, inertia))
            for mass, com, inertia in (raw_inertials[name] for name in JOINTS)
        ]
        self._build()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        ca = self._ca
        q, v, a, tau, p = self._q, self._v, self._a, self._tau, self._p
        n = self.n_q

        # Forward kinematics: world position/rotation of every joint origin.
        positions = []
        rotations = []
        pos = ca.SX.zeros(3)
        rot = ca.SX.eye(3)
        for i in range(n):
            pos = pos + rot @ self._offsets[i]
            rot = rot @ _rotation(ca, self._axes[i], q[i])
            positions.append(pos)
            rotations.append(rot)
        joint_positions = ca.horzcat(*positions)
        self._joint_positions_expr = joint_positions

        marker_columns = []
        for name in MARKER_NAMES:
            if name == "clubface":
                # Rotated by the terminal DOF, so shaft roll is observable.
                marker_columns.append(
                    positions[-1] + rotations[-1] @ ca.SX(list(CLUBFACE_OFFSET))
                )
            else:
                marker_columns.append(positions[JOINTS.index(_MARKER_JOINT[name])])
        markers = ca.horzcat(*marker_columns)
        markers_velocity = ca.jtimes(markers, q, v)

        total_mass: Any = 0.0
        weighted: Any = ca.SX.zeros(3)
        for i in range(n):
            mass, com, _inertia = self._inertials[i]
            total_mass = total_mass + mass
            weighted = weighted + mass * (positions[i] + rotations[i] @ com)
        com_world = weighted / total_mass

        tau_rnea = rnea_expression(
            ca, q, v, a, self._offsets, self._axes, self._inertials
        )
        self._rnea_expr = tau_rnea
        rnea = ca.Function("swing_rnea", [q, v, a, p], [tau_rnea])
        zero = ca.SX.zeros(n)
        bias = rnea(q, zero, zero, p)
        columns = [rnea(q, zero, ca.SX.eye(n)[:, i], p) - bias for i in range(n)]
        mass_matrix = ca.horzcat(*columns)
        h = rnea(q, v, zero, p)
        qddot = ca.solve(mass_matrix, tau - h)

        clubhead = positions[-1]
        clubhead_velocity = ca.jtimes(clubhead, q, v)

        self.rnea = rnea
        self.mass_matrix = ca.Function("swing_mass_matrix", [q, p], [mass_matrix])
        self.nonlinear_effects = ca.Function("swing_nonlinear_effects", [q, v, p], [h])
        self.forward_dynamics = ca.Function(
            "swing_forward_dynamics", [q, v, tau, p], [qddot]
        )
        self.joint_positions = ca.Function(
            "swing_joint_positions", [q, p], [joint_positions]
        )
        self.markers = ca.Function("swing_markers", [q, p], [markers])
        self.markers_velocities = ca.Function(
            "swing_markers_velocities", [q, v, p], [markers_velocity]
        )
        self.clubhead_position = ca.Function("swing_clubhead", [q, p], [clubhead])
        self.clubhead_velocity = ca.Function(
            "swing_clubhead_velocity", [q, v, p], [clubhead_velocity]
        )
        self.center_of_mass = ca.Function("swing_com", [q, p], [com_world])
        self.center_of_mass_velocity = ca.Function(
            "swing_com_velocity", [q, v, p], [ca.jtimes(com_world, q, v)]
        )
        self.total_mass = ca.Function("swing_total_mass", [p], [total_mass])
        self.gravity = ca.Function("swing_gravity", [p], [ca.SX(_GRAVITY)])

    # -- convenience ---------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    @property
    def n_markers(self) -> int:
        return len(self.marker_names)

    def marker_index(self, name: str) -> int:
        """Column of ``name`` in :attr:`markers` output."""
        try:
            return self.marker_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def nominal_parameters(self) -> np.ndarray:
        """Values of the declared parameters that reproduce the numeric model."""
        nominal = {
            "arm_length": self.golfer.arm_length,
            "trunk_length": self.golfer.trunk_length,
            "club_length": self.club.total_length,
            "mass": self.golfer.mass,
            "head_mass": self.club.head_mass,
        }
        return np.array([nominal[name] for name in self.parameter_names], dtype=float)

    def parameter_symbols(self) -> Any:
        """The ``SX`` parameter vector every function takes as its last input."""
        return self._p

    def torque_limits(self) -> np.ndarray:
        """Per-DOF torque bounds from the golfer, in ``JOINTS`` order."""
        golfer = self.golfer
        limits = {
            "hip_rotation": golfer.max_hip_torque,
            "trunk_rotation": golfer.max_trunk_torque,
            "shoulder_horizontal": golfer.max_shoulder_torque,
            "shoulder_vertical": golfer.max_shoulder_torque,
            "elbow_flexion": golfer.max_elbow_torque,
            "wrist_cock": golfer.max_wrist_torque,
            "wrist_rotation": golfer.max_wrist_torque,
        }
        return np.array([limits[name] for name in JOINTS], dtype=float)
