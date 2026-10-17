"""IK targets and external loads of a LIFT-5 bundle in the OpenSim frame (OSL-3).

A same-input lift bundle (LIFT-5, #11745) replays the full-body golf spec,
not the Rajagopal model OpenSim lifts use, so the bundle's coordinates do not
transfer by name (its ``hip_flexion_r`` reads 160-171 deg in the back-squat
window, outside Rajagopal's -30..120 deg range; its left knee is negative).
What does transfer is geometry and load:

* **Landmarks.**  The shared MuJoCo marker kinematics of the bundle spec
  (:class:`FullBodyMarkerKinematics`, the attachments the bundle's own IK
  tracked: ankles, knees, hips, shoulders) plus heel and forefoot points on
  the calcanei (the spec's own contact-sphere stations) and the bar centre of
  mass from the barbell provenance (LIFT-4, #11744).
* **Foot wrench (GCV-1, #11707).**  The bundle reference engine's shared-law
  contact at each state, reduced per foot by
  :func:`~src.shared.python.lifting.kinetics_parity.bundle_kinetics`.
* **Hand wrench (GCV-7, #11713).**  The per-hand bushing wrench on the bar
  (:meth:`BushingSample.grip_analysis`); the hands carry its negative.
* **Pelvis frame (OSL-12, #13048).**  The spec grafts Rajagopal lower limbs
  onto the golf ``LowerTorso``; both hip joints share one parent orientation,
  and that orientation, with the spec's hip axis permutation
  (``spec_builder.HIP_PERMUTATION``), is the Rajagopal pelvis frame:
  :func:`bundle_pelvis_rotation`.  Its source hip coordinates are kept so a
  caller can check them against a model's range.

Everything is re-expressed in the OpenSim lift frame: ``x`` forward (the
bundle's stance direction, heel to forefoot), ``y`` up, ``z`` to the
lifter's right, origin on the floor under mid-foot of the first state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.lifting.bundles import (
    LANDMARK_ATTACHMENTS,
    stance_frame,
    stance_mid_foot,
)
from src.shared.python.lifting.sagittal_retarget import LANDMARK_LABELS
from src.shared.python.motion_matching.same_input import InputBundle

Array = NDArray[np.float64]

__all__ = [
    "FOOT_LANDMARKS",
    "PELVIS_BODY",
    "SOURCE_HIP_COORDINATES",
    "bundle_pelvis_rotation",
    "IK_LANDMARKS",
    "LiftTargets",
    "LiftFrame",
    "lift_targets",
]

#: Calcaneus points tracked so the foot orientation (ankle angle) is observed.
FOOT_LANDMARKS: tuple[str, ...] = ("heel_l", "heel_r", "forefoot_l", "forefoot_r")
IK_LANDMARKS: tuple[str, ...] = LANDMARK_LABELS + FOOT_LANDMARKS
_FEET = {"left": "l", "right": "r"}
#: The spec body the Rajagopal hips are grafted onto (the golf pelvis).
PELVIS_BODY = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
#: Source hip coordinates kept per state (Rajagopal names and conventions).
SOURCE_HIP_COORDINATES: tuple[str, ...] = (
    "hip_flexion_l",
    "hip_flexion_r",
    "hip_adduction_l",
    "hip_adduction_r",
    "hip_rotation_l",
    "hip_rotation_r",
)
_HIP_JOINTS = ("hip_l", "hip_r")
_HANDS = ("L", "R")


@dataclass(frozen=True)
class LiftFrame:
    """Rigid map from the bundle world to the OpenSim lift frame.

    ``rotation`` rows are the bundle-world directions of OpenSim ``x``
    (forward), ``y`` (up) and ``z`` (right); ``origin_m`` is the bundle-world
    point that becomes the OpenSim origin.
    """

    rotation: Array
    origin_m: Array

    def point(self, p: Array) -> Array:
        """Bundle-world points (..., 3) in the OpenSim frame."""
        return (np.asarray(p, dtype=float) - self.origin_m) @ self.rotation.T

    def vector(self, v: Array) -> Array:
        """Bundle-world free vectors (..., 3) in the OpenSim frame."""
        return np.asarray(v, dtype=float) @ self.rotation.T

    def moment_about_origin(self, moment: Array, force: Array) -> Array:
        """A moment about the bundle origin, moved to the OpenSim origin."""
        shifted = np.asarray(moment, float) - np.cross(self.origin_m, force)
        return self.vector(shifted)

    def record(self) -> dict[str, Any]:
        return {
            "axes": "x forward (heel to forefoot), y up, z right",
            "rotation_rows": self.rotation.tolist(),
            "origin_in_bundle_world_m": self.origin_m.tolist(),
        }


@dataclass(frozen=True)
class LiftTargets:
    """Landmark targets and external loads of a bundle, OpenSim frame.

    Arrays are per selected state ``indices``: ``positions_m`` (N, M, 3) in
    :data:`IK_LANDMARKS` order; ``grf_force_n`` / ``grf_moment_nm`` map
    ``left``/``right`` to (N, 3) foot force and moment about the OpenSim
    origin; ``hand_force_n`` / ``hand_torque_nm`` map ``L``/``R`` to the
    (N, 3) force and free torque the bar exerts on each hand;
    ``pelvis_rotation`` (N, 3, 3) holds the bundle's Rajagopal pelvis axes
    (columns) in the OpenSim frame; ``source_coordinates`` maps each
    :data:`SOURCE_HIP_COORDINATES` name to its (N,) bundle values (radians).
    """

    lift: str
    path: Path
    sha256: str
    indices: tuple[int, ...]
    time_s: Array
    positions_m: Array
    foot_stations_m: Mapping[str, tuple[str, tuple[float, float, float]]]
    grf_force_n: Mapping[str, Array]
    grf_moment_nm: Mapping[str, Array]
    hand_force_n: Mapping[str, Array]
    hand_torque_nm: Mapping[str, Array]
    pelvis_rotation: Array
    source_coordinates: Mapping[str, Array]
    body_mass_kg: float
    bar_mass_kg: float
    height_m: float
    frame: LiftFrame
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def landmarks(self, k: int) -> dict[str, Array]:
        """Landmark positions of state ``k`` (of the selection) by label."""
        return dict(zip(IK_LANDMARKS, self.positions_m[k], strict=True))


def _foot_stations(spec: Mapping[str, Any]) -> dict[str, tuple[str, tuple]]:
    """Heel and forefoot stations on the calcanei from the spec's spheres."""
    spheres = {s["name"]: s for s in spec["contact"]["spheres"]}
    stations = {}
    for label in FOOT_LANDMARKS:
        if label not in spheres:
            raise ValueError(f"bundle spec has no contact sphere {label!r}")
        sphere = spheres[label]
        stations[label] = (str(sphere["body"]), tuple(map(float, sphere["position_m"])))
    return stations


def bundle_pelvis_rotation(spec: Mapping[str, Any]) -> Array:
    """Rajagopal pelvis axes expressed in the spec's pelvis body frame.

    Both hip joints hang from one pelvis.  The right hip's base frame is the
    Rajagopal hip frame with the spec's axis permutation, so
    ``R_base @ HIP_PERMUTATION.T`` gives the pelvis axes (columns) in
    :data:`PELVIS_BODY` coordinates.  The left hip's base must be the same
    orientation, or (the AN-1 lift body, OSL-14 #13057) that orientation
    conjugated by ``spec_builder.HIP_MIRROR``, which mirrors the left leg while
    keeping OpenSim's coordinate signs.

    Postcondition: the result is a proper rotation (orthonormal, det +1).

    Raises:
        ValueError: when a hip joint is missing, hangs from another body, or
            the left hip base is neither the right one nor its mirror.
    """
    from src.shared.python.motion_matching.execution.spec_builder import (
        HIP_MIRROR,
        HIP_PERMUTATION,
    )

    joints = {j["name"]: j for j in spec["joints"]}
    bases = []
    for name in _HIP_JOINTS:
        joint = joints.get(name)
        if joint is None or joint["parent"] != PELVIS_BODY:
            raise ValueError(f"spec joint {name!r} does not hang from the pelvis")
        bases.append(np.asarray(joint["parent_to_base"], dtype=float)[:3, :3])
    left, right = bases
    mirror = np.asarray(HIP_MIRROR, dtype=float)[:3, :3]
    if not (
        np.allclose(left, right, atol=1e-12)
        or np.allclose(left, right @ mirror, atol=1e-12)
    ):
        raise ValueError("the spec's hip joints do not share a pelvis orientation")
    rotation = right @ HIP_PERMUTATION.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9) or (
        np.linalg.det(rotation) <= 0.0
    ):
        raise ValueError("the spec's hip parent frame is not a proper rotation")
    return rotation


def _kinematics(spec_bytes: bytes, attachments: Mapping[str, Any]) -> tuple[Any, float]:
    """Shared MuJoCo marker kinematics of the spec and its total mass."""
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    adapter = NativeMujocoFullBodyModel(bytes(spec_bytes))
    mass = float(np.sum(adapter.model.body_mass))
    return FullBodyMarkerKinematics(adapter, dict(attachments)), mass


def _lift_frame(kin: Any, q0: Array) -> LiftFrame:
    from src.shared.python.motion_matching.contact_law import GroundPlane

    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    sagittal = stance_frame(kin, q0, ground)
    forward = np.asarray(sagittal.forward)
    rotation = np.vstack([forward, np.asarray(sagittal.up), -sagittal.left])
    return LiftFrame(rotation=rotation, origin_m=stance_mid_foot(kin, q0, ground))


def _hand_loads(
    bundle: InputBundle, indices: tuple[int, ...], frame: LiftFrame
) -> tuple[dict[str, Array], dict[str, Array], Array]:
    """GCV-7 bar wrench on each hand and the bar centre, per state."""
    from src.shared.python.lifting.barbell_runs import (
        augmented_reference,
        barbell_plant,
    )

    plant = barbell_plant(bundle.reference_engine, bundle)
    q_ref, v_ref = augmented_reference(bundle)
    forces: dict[str, list[Array]] = {side: [] for side in _HANDS}
    torques: dict[str, list[Array]] = {side: [] for side in _HANDS}
    bar = []
    for k in indices:
        sample = plant.sample(q_ref[k], v_ref[k])
        grip = sample.grip_analysis()
        for side, hand in zip(_HANDS, (grip.left, grip.right), strict=True):
            if hand is None or hand.torque_on_club_nm is None:
                raise ValueError(f"bundle state {k}: hand {side} wrench unavailable")
            forces[side].append(-np.asarray(hand.force_on_club_n, float))
            torques[side].append(-np.asarray(hand.torque_on_club_nm, float))
        bar.append(np.asarray(sample.bar.position_m, float))
    return (
        {s: frame.vector(np.array(forces[s])) for s in _HANDS},
        {s: frame.vector(np.array(torques[s])) for s in _HANDS},
        frame.point(np.array(bar)),
    )


def _foot_loads(
    bundle: InputBundle, indices: tuple[int, ...], frame: LiftFrame
) -> tuple[dict[str, Array], dict[str, Array]]:
    """GCV-1 foot force and moment about the OpenSim origin, per state."""
    from src.shared.python.lifting.kinetics_parity import bundle_kinetics

    samples = bundle_kinetics(bundle.reference_engine, bundle, indices)
    forces, moments = {}, {}
    for foot in _FEET:
        force = np.array([s.per_foot[foot].force_n for s in samples])
        moment = np.array([s.per_foot[foot].moment_about_origin_nm for s in samples])
        forces[foot] = frame.vector(force)
        moments[foot] = np.array(
            [
                frame.moment_about_origin(m, f)
                for m, f in zip(moment, force, strict=True)
            ]
        )
    return forces, moments


def lift_targets(lift: str, path: Path, *, stride: int = 1) -> LiftTargets:
    """Landmarks and external loads of a barbell same-input bundle.

    Args:
        lift: Lift name, recorded in the receipt.
        path: The bundle ``.npz`` (it must carry the LIFT-4 barbell record).
        stride: Use every ``stride``-th reference state (first and last kept).

    Postconditions: ``positions_m`` holds every :data:`IK_LANDMARKS` entry for
    every selected state; ``sha256`` is the digest of the bundle file bytes.

    Raises:
        ValueError: for a non-positive stride, a bundle without the barbell
            record, or a spec without heel and forefoot spheres.
    """
    if stride < 1:
        raise ValueError(f"stride must be a positive integer, got {stride}")
    raw = Path(path).read_bytes()
    bundle = InputBundle.load(Path(path))
    record = bundle.provenance.get("barbell")
    if not isinstance(record, Mapping):
        raise ValueError("bundle carries no barbell record (LIFT-4 bar states)")
    spec = json.loads(bundle.spec_bytes)
    stations = _foot_stations(spec)
    attachments = {
        label: LANDMARK_ATTACHMENTS[label] for label in LANDMARK_LABELS
    } | stations
    kin, body_mass = _kinematics(bundle.spec_bytes, attachments)
    frame = _lift_frame(kin, bundle.reference_q[0])
    indices = tuple(sorted({*range(0, bundle.steps + 1, stride), bundle.steps}))
    order = tuple(attachments)
    points = np.array([kin.marker_positions(bundle.reference_q[k]) for k in indices])
    positions = frame.point(points[:, [order.index(x) for x in IK_LANDMARKS]])
    in_pelvis = bundle_pelvis_rotation(spec)
    pelvis = np.array(
        [
            frame.rotation
            @ kin.body_poses(bundle.reference_q[k], [PELVIS_BODY])[PELVIS_BODY][0]
            @ in_pelvis
            for k in indices
        ]
    )
    order_q = list(spec["coordinate_order"])
    source = {
        name: np.array([bundle.reference_q[k][order_q.index(name)] for k in indices])
        for name in SOURCE_HIP_COORDINATES
    }
    hand_force, hand_torque, bar = _hand_loads(bundle, indices, frame)
    positions[:, IK_LANDMARKS.index("bar")] = bar
    grf_force, grf_moment = _foot_loads(bundle, indices, frame)
    return LiftTargets(
        lift=lift,
        path=Path(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        indices=indices,
        time_s=np.asarray(indices, dtype=float) * float(bundle.dt_s),
        positions_m=positions,
        foot_stations_m=stations,
        grf_force_n=grf_force,
        grf_moment_nm=grf_moment,
        hand_force_n=hand_force,
        hand_torque_nm=hand_torque,
        pelvis_rotation=pelvis,
        source_coordinates=source,
        body_mass_kg=body_mass,
        bar_mass_kg=float(record["bar_mass_kg"]),
        height_m=float(bundle.provenance["optimizer"]["height_m"]),
        frame=frame,
        provenance={
            "reference_engine": bundle.reference_engine,
            "dt_s": float(bundle.dt_s),
            "steps": int(bundle.steps),
            "window": dict(bundle.provenance.get("window", {})),
            "spec_sha256": bundle.spec_sha256,
        },
    )
