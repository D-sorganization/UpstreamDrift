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
from typing import Final

import numpy as np

from src.engines.physics_engines.mujoco.python.native_model import (
    _evaluate_weld_closure,
)
from src.engines.physics_engines.mujoco.python.overlay_source import (
    MujocoOverlaySource,
)
from src.shared.python.engine_core.mujoco_compat import full_mass_matrix
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.grip_hands import grip_hand_frame
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.spatial_algebra.operational_space import (
    force_ellipsoid,
    mobility_matrix,
    velocity_ellipsoid,
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


class HandMobilitySource(MujocoOverlaySource):
    """``MujocoOverlaySource`` that also reports per-hand mobility."""

    def __init__(self, model_bytes: bytes, pose: GripPose = DEFAULT_GRIP_POSE) -> None:
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
            f"taken on body {names[1]}"
        )

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

    def _set_state(self, q: np.ndarray) -> dict[str, float]:
        values = np.asarray(q, dtype=float)
        if (
            values.shape != (len(self.coordinate_order),)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"q must be {len(self.coordinate_order)} finite coordinates"
            )
        coordinates = dict(zip(self.coordinate_order, values.tolist(), strict=True))
        self._forward(coordinates)
        return coordinates

    def _mobility_at(
        self, point: np.ndarray, rows: int, constrained: bool
    ) -> np.ndarray:
        mj, model, data = self._mj, self.model, self.data
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jac(model, data, jacp, jacr, point, self._club_body)
        jac = jacp if rows == 3 else np.vstack((jacp, jacr))
        mass = full_mass_matrix(mj, model, data)
        closure = _evaluate_weld_closure(mj, model, data, self._closure)[0]
        return mobility_matrix(jac, mass, closure if constrained else None)

    def hand_mobility(
        self, q: np.ndarray, *, mode: str = "constrained"
    ) -> dict[str, HandMobility]:
        """Lead and trail 3x3 mobility at state ``q`` (``coordinate_order``)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        coordinates = self._set_state(q)
        armature = np.asarray(self.model.dof_armature, dtype=float)
        provenance = {
            "topology": self.topology,
            "armature_kg_m2": [float(armature.min()), float(armature.max())],
            "frame": "world",
            "mode": mode,
            "point": "grip_hand_frame origin on the club",
        }
        out: dict[str, HandMobility] = {}
        for hand in HANDS:
            point = self.grip_point_world(hand, coordinates)
            w = self._mobility_at(point, 3, mode == "constrained")
            lam = velocity_ellipsoid(w)
            dual = force_ellipsoid(w)
            eig = np.maximum(lam.radii**2, 0.0)
            mass_eff = dual.radii**2
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
            )
        return out
