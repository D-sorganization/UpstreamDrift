"""Per-hand operational-space mobility on the MuJoCo golfer (HCM-3, #14004).

For each hand the 3x3 translational mobility ``W = J A J^T`` [kg^-1] at that
hand's grip point on the club: ``J`` is the *absolute* point-velocity Jacobian
(``mj_jac``), ``A`` the inverse mass matrix (``Jc=None``, "loop released") or
the closure projector (``constrained``, both hands on the club) from
``spatial_algebra.operational_space.mobility_matrix``.  The mass matrix is
MuJoCo's full ``M`` including dof armature; the closure Jacobian is the one
the native model defines (``_evaluate_weld_closure``).  The reciprocal
eigenvalues are *effective masses* [kg], an inertial property, not a force
capacity.  Only the explicit rigid closure is used, never the compliant MJCF
weld equality.

Topology: in the default v2 golfer the lead hand is part of the club body, so
both grip points lie on one rigid body and the constrained spatial mobilities
there are related by ``transport_spatial_mobility``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from src.engines.physics_engines.mujoco.python.native_model import (
    _evaluate_weld_closure,
)
from src.engines.physics_engines.mujoco.python.overlay_source import (
    MujocoOverlaySource,
)
from src.shared.python.engine_core.mujoco_compat import full_mass_matrix
from src.shared.python.loop_nullspace.capacity import CapacitySubject
from src.shared.python.loop_nullspace.torque_limits import (
    TorqueLimits,
    torque_limits_for,
)
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.grip_hands import grip_hand_frame
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.spatial_algebra.operational_space import (
    CapacityEllipsoid,
    acceleration_capacity,
    force_capacity,
    force_ellipsoid,
    mobility_matrix,
    velocity_ellipsoid,
)
from src.shared.python.spatial_algebra.operational_space import (
    drift_acceleration as _calc_drift_acceleration,
)

MODES: Final = ("constrained", "unconstrained")
HANDS: Final = ("lead", "trail")

__all__ = ["HANDS", "MODES", "HandMobility", "HandMobilitySource"]


@dataclass(frozen=True)
class HandMobility:
    """Mobility of one hand at one frame (world frame).

    ``mobility`` is W [kg^-1]; ``mobility_axes`` its eigenvectors (columns,
    ascending eigenvalue), ``mobility_eigenvalues`` the eigenvalues,
    ``effective_mass_kg`` the 1/lambda per axis (clamped where ``singular``),
    ``condition`` lambda_max/lambda_min.  ``provenance`` records topology,
    armature, frame, mode.
    """

    hand: str
    centre: tuple[float, float, float]
    mobility: np.ndarray
    mobility_axes: np.ndarray
    mobility_eigenvalues: np.ndarray
    effective_mass_kg: np.ndarray
    condition: float
    singular: np.ndarray
    provenance: Mapping[str, object]
    acceleration_capacity: CapacityEllipsoid | None = None
    force_capacity: CapacityEllipsoid | None = None
    drift_acceleration: np.ndarray | None = None
    jdot_v: np.ndarray | None = None


class HandMobilitySource(MujocoOverlaySource):
    """``MujocoOverlaySource`` that also reports per-hand mobility and capacity."""

    def __init__(
        self,
        model_bytes: bytes,
        pose: GripPose = DEFAULT_GRIP_POSE,
        torque_limits: TorqueLimits | CapacitySubject | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model_bytes)
        spec = json.loads(model_bytes)
        club = assembly_from_spec(spec)
        if club is None:
            raise ValueError("the specification has no club for the grip points")
        self._grip_in_club = {
            hand: grip_hand_frame(club, Hand(hand), pose)[:3, 3] for hand in HANDS
        }
        mj, model = self._mj, self.model
        names = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.site_bodyid[s]))
            for s in self._closure
        ]
        self._club_body = int(model.site_bodyid[self._closure[1]])
        self.topology = (
            f"rigid 6-D closure {names[0]} <-> {names[1]} (explicit KKT; the MJCF "
            "weld equality is compliant and not used); both grip points are "
            f"taken on body {names[1]} (welded lead hand topology: lead hand is "
            "welded to the club in v2 golfer, so lead and trail share one rigid body)"
        )
        self._torque_limits = self._resolve_torque_limits(torque_limits, spec)

    def _resolve_torque_limits(
        self,
        source: TorqueLimits | CapacitySubject | Mapping[str, Any] | None,
        spec: Mapping[str, Any],
    ) -> TorqueLimits | None:
        if isinstance(source, TorqueLimits):
            return source
        if isinstance(source, CapacitySubject):
            return torque_limits_for(source, self.coordinate_order)
        if isinstance(source, Mapping):
            return torque_limits_for(source, self.coordinate_order)
        if "subject" in spec:
            try:
                return torque_limits_for(spec, self.coordinate_order)
            except Exception:
                return None
        return None

    def grip_point_world(
        self, hand: str, coordinates: Mapping[str, float]
    ) -> np.ndarray:
        """World grip point of ``hand`` at ``coordinates`` (club frame + grip)."""
        if hand not in self._grip_in_club:
            raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
        rot, origin = self.club_frame_world(coordinates)
        return origin + rot @ self._grip_in_club[hand]

    def spatial_mobility(
        self, q: np.ndarray, hand: str, *, constrained: bool = True
    ) -> np.ndarray:
        """6x6 [linear; angular] mobility at ``hand``'s grip point (unscaled)."""
        coordinates = self._set_state(q)
        point = self.grip_point_world(hand, coordinates)
        return self._mobility_at(point, 6, constrained)

    def _set_state(
        self, q: np.ndarray, v: np.ndarray | None = None
    ) -> dict[str, float]:
        values = np.asarray(q, dtype=float)
        if (
            values.shape != (len(self.coordinate_order),)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"q must be {len(self.coordinate_order)} finite coordinates"
            )
        coordinates = dict(zip(self.coordinate_order, values.tolist(), strict=True))
        rates: dict[str, float] | None = None
        if v is not None:
            v_values = np.asarray(v, dtype=float)
            if (
                v_values.shape != (len(self.coordinate_order),)
                or not np.isfinite(v_values).all()
            ):
                raise ValueError(
                    f"v must be {len(self.coordinate_order)} finite velocities"
                )
            rates = dict(zip(self.coordinate_order, v_values.tolist(), strict=True))
        self._forward(coordinates, rates)
        return coordinates

    def bias_force(
        self,
        coordinates: Mapping[str, float] | np.ndarray,
        rates: Mapping[str, float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Generalised bias force h(q, v) [N or N*m] (length nv)."""
        if isinstance(coordinates, Mapping):
            q_vec = np.array(
                [coordinates[name] for name in self.coordinate_order], dtype=float
            )
        else:
            q_vec = np.asarray(coordinates, dtype=float)
        if rates is not None:
            if isinstance(rates, Mapping):
                v_vec: np.ndarray | None = np.array(
                    [rates[name] for name in self.coordinate_order], dtype=float
                )
            else:
                v_vec = np.asarray(rates, dtype=float)
        else:
            v_vec = None
        coords = self._set_state(q_vec, v_vec)
        r_map = (
            dict(
                zip(
                    self.coordinate_order,
                    v_vec.tolist(),
                    strict=True,
                )
            )
            if v_vec is not None
            else None
        )
        return super().bias_force(coords, r_map)

    def grip_jdot_v(self, hand: str, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """3-vector J̇(q, v) v [m/s^2] at ``hand``'s grip point on the club body."""
        coords = self._set_state(q, v)
        point = self.grip_point_world(hand, coords)
        return self.point_jdot_v(point, self._club_body, coords)

    def drift_acceleration(
        self,
        q: np.ndarray,
        v: np.ndarray,
        hand: str,
        *,
        mode: str = "constrained",
    ) -> np.ndarray:
        """3-vector drift acceleration c = J a0 + J̇v at ``hand``'s grip point [m/s^2]."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if hand not in HANDS:
            raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
        coords = self._set_state(q, v)
        point = self.grip_point_world(hand, coords)
        mj, model, data = self._mj, self.model, self.data
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jac(model, data, jacp, jacr, point, self._club_body)
        mass = full_mass_matrix(mj, model, data)
        h = np.asarray(data.qfrc_bias, dtype=float).copy()
        jdot_pos = np.zeros((3, model.nv))
        jdot_rot = np.zeros((3, model.nv))
        mj.mj_jacDot(model, data, jdot_pos, jdot_rot, point, self._club_body)
        jdot_v = jdot_pos @ data.qvel

        if mode == "constrained":
            closure_jac, closure_drift = _evaluate_weld_closure(
                mj, model, data, self._closure
            )
        else:
            closure_jac, closure_drift = None, None

        return _calc_drift_acceleration(
            jacp, mass, h, jdot_v, Jc=closure_jac, Jcdot_v=closure_drift
        )

    def _task_matrices(
        self, point: np.ndarray, constrained: bool, rows: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        mj, model, data = self._mj, self.model, self.data
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jac(model, data, jacp, jacr, point, self._club_body)
        jac = jacp if rows == 3 else np.vstack((jacp, jacr))
        mass = full_mass_matrix(mj, model, data)
        closure = (
            _evaluate_weld_closure(mj, model, data, self._closure)[0]
            if constrained
            else None
        )
        return jac, mass, closure

    def _mobility_at(
        self, point: np.ndarray, rows: int, constrained: bool
    ) -> np.ndarray:
        jac, mass, closure = self._task_matrices(point, constrained, rows=rows)
        return mobility_matrix(jac, mass, closure)

    def _capacity_task(
        self,
        q: np.ndarray,
        hand: str,
        mode: str,
        torque_limits: TorqueLimits | None,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, TorqueLimits]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if hand not in HANDS:
            raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
        limits = torque_limits or self._torque_limits
        if limits is None:
            raise ValueError(f"{label} capacity requires torque limits")
        coordinates = self._set_state(q)
        point = self.grip_point_world(hand, coordinates)
        J, M, Jc = self._task_matrices(point, mode == "constrained")
        return J, M, Jc, limits

    def acceleration_capacity(
        self,
        q: np.ndarray,
        hand: str,
        *,
        mode: str = "constrained",
        torque_limits: TorqueLimits | None = None,
    ) -> CapacityEllipsoid:
        """Acceleration-capacity ellipsoid at ``hand``'s grip point [m/s^2]."""
        J, M, Jc, limits = self._capacity_task(
            q, hand, mode, torque_limits, "acceleration"
        )
        return acceleration_capacity(J, M, limits.B, limits.tau_lo, limits.tau_hi, Jc)

    def force_capacity(
        self,
        q: np.ndarray,
        hand: str,
        *,
        mode: str = "constrained",
        torque_limits: TorqueLimits | None = None,
    ) -> CapacityEllipsoid:
        """Dynamically consistent force-capacity ellipsoid at ``hand``'s grip point [N]."""
        J, M, Jc, limits = self._capacity_task(q, hand, mode, torque_limits, "force")
        return force_capacity(J, M, limits.B, limits.tau_lo, limits.tau_hi, Jc)

    def hand_mobility(
        self,
        q: np.ndarray,
        *,
        v: np.ndarray | None = None,
        mode: str = "constrained",
        torque_limits: TorqueLimits | None = None,
    ) -> dict[str, HandMobility]:
        """Lead and trail 3x3 mobility and capability at state ``q`` (``coordinate_order``)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        limits = torque_limits or self._torque_limits
        coordinates = self._set_state(q, v)
        armature = np.asarray(self.model.dof_armature, dtype=float)
        provenance = {
            "topology": self.topology,
            "armature_kg_m2": [float(armature.min()), float(armature.max())],
            "frame": "world",
            "mode": mode,
            "point": "grip_hand_frame origin on the club",
            "welded_lead_hand": True,
            "caption": (
                "Lead hand is welded to the club in v2 golfer topology; "
                "lead and trail share rigid body kinematics on the club."
            ),
            "torque_limit_source": limits.source_id if limits is not None else None,
            "assumed_slots": limits.assumed_slots if limits is not None else 0,
            "verified_slots": limits.verified_slots if limits is not None else 0,
        }
        out: dict[str, HandMobility] = {}
        for hand in HANDS:
            point = self.grip_point_world(hand, coordinates)
            w = self._mobility_at(point, 3, mode == "constrained")
            lam = velocity_ellipsoid(w)
            dual = force_ellipsoid(w)
            eig = np.maximum(lam.radii**2, 0.0)
            mass_eff = dual.radii**2

            acc_cap: CapacityEllipsoid | None = None
            f_cap: CapacityEllipsoid | None = None
            if limits is not None:
                J, M, Jc = self._task_matrices(point, mode == "constrained")
                acc_cap = acceleration_capacity(
                    J, M, limits.B, limits.tau_lo, limits.tau_hi, Jc
                )
                f_cap = force_capacity(J, M, limits.B, limits.tau_lo, limits.tau_hi, Jc)

            drift_acc: np.ndarray | None = None
            jdot_v: np.ndarray | None = None
            if v is not None:
                drift_acc = self.drift_acceleration(q, v, hand, mode=mode)
                jdot_v = self.grip_jdot_v(hand, q, v)

            out[hand] = HandMobility(
                hand=hand,
                centre=(float(point[0]), float(point[1]), float(point[2])),
                mobility=w,
                mobility_axes=lam.axes,
                mobility_eigenvalues=eig,
                effective_mass_kg=mass_eff,
                condition=float(mass_eff.max() / mass_eff.min()),
                singular=dual.singular,
                provenance=provenance,
                acceleration_capacity=acc_cap,
                force_capacity=f_cap,
                drift_acceleration=drift_acc,
                jdot_v=jdot_v,
            )
        return out
