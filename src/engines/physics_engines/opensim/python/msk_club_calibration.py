"""Address calibration of the two-hand club grip on the Rajagopal models (OSV-9).

The captured address pose reaches the musculoskeletal models through the
generated full-body model, which is matched to the same capture (capture A
for the driver, capture B for the 7-iron; ``tests/fixtures/club_face/
address_poses.json``):

1. The generated model's own OpenSim forward kinematics at its address pose
   gives the club pose and body landmarks (hip, knee, ankle, shoulder and
   elbow centres) in the native world (Z up, golfer facing -X, target -Y).
2. The native world maps to the Rajagopal world (Y up, golfer facing +X) by
   :data:`NATIVE_TO_OPENSIM`; the native floor (lowest head-mesh point at
   address, where the sole rests) maps to y = 0.
3. Inverse kinematics over the Rajagopal coordinates (bounded by their
   ranges) puts each hand's :data:`msk_club.HAND_GRIP_POINT_M` on its grip
   point on the shaft (2 mm weight), lays the shaft diagonally across each
   palm toward the thumb (0.1 weight), keeps the wrists in their
   physiological deviation range and follows the landmarks (4 cm weight), with a
   weak pull toward the neutral pose.
4. Each hand-side grip frame is the club grip frame seen from that hand at
   the solution, so both welds are exact at the address pose and the club
   sits exactly where the generated model holds it: the shared face roll
   then squares the face, which the tests check by OpenSim FK.

This module needs OpenSim and SciPy; :mod:`msk_club` reads its output.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import club_visuals
from src.engines.physics_engines.opensim.python import msk_club as mc
from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import club_face as cf

logger = logging.getLogger(__name__)

#: Rotation taking native-world vectors (Z up, facing -X, target -Y) to the
#: Rajagopal world (Y up, facing +X, target -Z).
NATIVE_TO_OPENSIM = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
ADDRESS_POSES = mc.REPO_ROOT / "tests" / "fixtures" / "club_face" / "address_poses.json"
#: Generated-model body -> Rajagopal body whose origins are the same joint
#: centre. The generated model's knees and ankles are not followed: its feet
#: are unobserved in the capture and leave the floor (ankles 0.14-0.5 m up,
#: 0.35 m behind the hips at address), so the feet are planted instead.
LANDMARKS = {
    "femur_r": "femur_r",
    "femur_l": "femur_l",
    "RS": "humerus_r",
    "LS": "humerus_l",
    "RE": "ulna_r",
    "LE": "ulna_l",
}
#: Planted feet: ankle centres this far either side of the address hip
#: midpoint along the target line, under the hips, feet flat and square.
STANCE_HALF_WIDTH_M = 0.22
#: Rajagopal foot body origins relative to the ankle centre with the foot
#: flat (x toward the ball, z toward the trail side); y is height above floor.
FOOT_TEMPLATE = {
    "talus": (0.0, 0.080, 0.0),
    "calcn": (-0.044, 0.038, 0.008),
    "toes": (0.118, 0.036, 0.009),
}
FOOT_WEIGHT_M = 0.01
GENERATED_CLUB_BODY = "Clubhead"
GRIP_WEIGHT_M = 0.002
THUMB_WEIGHT = 0.1
LANDMARK_WEIGHT_M = 0.04
NEUTRAL_WEIGHT_RAD = 2.0
#: Angle of the shaft from the radial axis toward the fingers: in a golf grip
#: the shaft lies diagonally across the palm, from the heel pad (ulnar,
#: proximal) to the index finger (radial, distal).
GRIP_DIAGONAL_DEG = 40.0
_DIAGONAL = np.radians(GRIP_DIAGONAL_DEG)
#: Direction of the head end of the shaft (club +y) in each hand frame: radial
#: (thumb, -z left / +z right) tilted toward the fingers (-y).
THUMB_AXIS = {
    "L": np.array([0.0, -np.sin(_DIAGONAL), -np.cos(_DIAGONAL)]),
    "R": np.array([0.0, -np.sin(_DIAGONAL), np.cos(_DIAGONAL)]),
}
#: Static address limits narrower than the golf models' swing ranges: the
#: Rajagopal physiological wrist deviation (the golf models widen it to
#: +/-45 deg for the swing, #10003), so the address pose also loads in the
#: muscle model, which keeps the physiological range.
ADDRESS_RANGES = {
    "wrist_dev_r": (-0.43633231, 0.61086524),
    "wrist_dev_l": (-0.43633231, 0.61086524),
}


def planted_feet(hips: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rajagopal foot-body targets under ``hips`` (``femur_r``/``femur_l``)."""
    middle = 0.5 * (np.asarray(hips["femur_r"]) + np.asarray(hips["femur_l"]))
    feet = {}
    for suffix, side in (("r", 1.0), ("l", -1.0)):
        for body, (x, y, z) in FOOT_TEMPLATE.items():
            centre_z = middle[2] + side * STANCE_HALF_WIDTH_M
            feet[f"{body}_{suffix}"] = np.array([middle[0] + x, y, centre_z + side * z])
    return feet


def landmark_weight(body: str) -> float:
    """Residual weight (m) of a landmark target on Rajagopal ``body``."""
    return FOOT_WEIGHT_M if body.split("_")[0] in FOOT_TEMPLATE else LANDMARK_WEIGHT_M


def _osim() -> Any:
    import opensim

    return opensim


def native_to_opensim(transform: np.ndarray, floor_z: float) -> np.ndarray:
    """4x4 native-world pose to the Rajagopal world (floor to y = 0)."""
    mat = np.asarray(transform, dtype=float)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    out = np.eye(4)
    out[:3, :3] = NATIVE_TO_OPENSIM @ mat[:3, :3]
    out[:3, 3] = NATIVE_TO_OPENSIM @ (mat[:3, 3] - np.array([0.0, 0.0, floor_z]))
    return out


def opensim_to_native_vector(vector: Sequence[float]) -> np.ndarray:
    """Rajagopal-world direction to the native world (inverse rotation)."""
    return NATIVE_TO_OPENSIM.T @ np.asarray(vector, dtype=float)


class PoseProbe:
    """Set coordinates and read body poses of a loaded OpenSim model (LOD wrapper)."""

    def __init__(self, model_path: Path) -> None:
        osim = _osim()
        club_visuals.register_geometry_path()
        self.model = osim.Model(str(model_path))
        self.state = self.model.initSystem()
        self._coords = self.model.getCoordinateSet()
        self._bodies = self.model.getBodySet()

    def set(self, values: Mapping[str, float]) -> None:
        """Set coordinate values (no constraint projection) and realise positions."""
        for name, value in values.items():
            self._coords.get(name).setValue(self.state, float(value), False)
        self.model.realizePosition(self.state)

    def body(self, name: str) -> np.ndarray:
        """4x4 pose of body ``name`` in ground."""
        return transform_in_ground(self._bodies.get(name), self.state)

    def coordinate_range(self, name: str) -> tuple[float, float]:
        coord = self._coords.get(name)
        return float(coord.getRangeMin()), float(coord.getRangeMax())

    def values(self) -> dict[str, float]:
        return {c.getName(): float(c.getValue(self.state)) for c in self._coords}


def transform_in_ground(frame: Any, state: Any) -> np.ndarray:
    """4x4 pose of an OpenSim frame in ground."""
    xform = frame.getTransformInGround(state)
    rot, pos = xform.R(), xform.p()
    out = np.eye(4)
    out[:3, :3] = [[rot.get(i, j) for j in range(3)] for i in range(3)]
    out[:3, 3] = [pos.get(i) for i in range(3)]
    return out


@dataclass(frozen=True)
class AddressTargets:
    """Club pose and landmark positions in the Rajagopal world at address."""

    club_in_ground: np.ndarray
    landmarks: dict[str, np.ndarray]
    floor_native_z: float


class GeneratedSwing:
    """The generated full-body model of ``club``, posed by its own OpenSim FK.

    Gives the club pose and landmarks in the Rajagopal world at any generated
    pose; the floor is fixed at the address pose (lowest head-mesh point).
    """

    def __init__(self, club: str) -> None:
        from src.engines.physics_engines.opensim.python.full_body_osim import (
            export_full_body_osim,
        )

        spec_bytes = mc.spec_path(club).read_bytes()
        address = json.loads(ADDRESS_POSES.read_text(encoding="utf-8"))["poses"][club]
        self.coordinate_order: list[str] = list(address["coordinate_order"])
        xml, _ = export_full_body_osim(spec_bytes, club_geometry_ref="")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generated.osim"
            path.write_text(xml, encoding="utf-8")
            self._probe = PoseProbe(path)
        assembly = ca.assembly_from_spec(json.loads(spec_bytes))
        if assembly is None:
            raise ValueError(f"{club} spec has no club assembly")
        self._head = ca.assembly_meshes(assembly)["head"].vertices
        self._probe.set(self._pose(address["q_rad"]))
        club_native = self._probe.body(GENERATED_CLUB_BODY)
        head = self._head @ club_native[:3, :3].T + club_native[:3, 3]
        self.floor_native_z = float(head[:, 2].min())
        self._feet = planted_feet(self._landmarks())

    def _landmarks(self) -> dict[str, np.ndarray]:
        floor = self.floor_native_z
        return {
            target: native_to_opensim(self._probe.body(source), floor)[:3, 3]
            for source, target in LANDMARKS.items()
        }

    def _pose(self, row: Sequence[float]) -> dict[str, float]:
        return dict(zip(self.coordinate_order, map(float, row), strict=True))

    def targets(self, row: Sequence[float]) -> AddressTargets:
        """Club pose and landmarks (Rajagopal world) at generated pose ``row``."""
        self._probe.set(self._pose(row))
        floor = self.floor_native_z
        landmarks = {**self._landmarks(), **self._feet}
        club = native_to_opensim(self._probe.body(GENERATED_CLUB_BODY), floor)
        return AddressTargets(club, landmarks, floor)


def generated_address_targets(club: str) -> AddressTargets:
    """Targets from the generated model's OpenSim FK at its address pose."""
    address = json.loads(ADDRESS_POSES.read_text(encoding="utf-8"))["poses"][club]
    return GeneratedSwing(club).targets(address["q_rad"])


def grip_frames(club: mc.MskClub, club_in_ground: np.ndarray) -> dict[str, np.ndarray]:
    frames = {}
    for side in mc.HAND_BODIES:
        grip = np.eye(4)
        grip[:3, :3] = club.grip_rotation
        grip[:3, 3] = club.grip_points[side]
        frames[side] = club_in_ground @ grip
    return frames


class _Residual:
    """Weighted IK residual over the free coordinates."""

    def __init__(
        self,
        probe: PoseProbe,
        names: list[str],
        targets: AddressTargets,
        grips: dict[str, np.ndarray],
    ) -> None:
        self.probe, self.names, self.targets, self.grips = probe, names, targets, grips
        self.shaft = targets.club_in_ground[:3, 1]
        self.landmarks = targets.landmarks
        self.rotational = np.array([not n.startswith("pelvis_t") for n in names])

    def __call__(self, q: np.ndarray) -> np.ndarray:
        self.probe.set(dict(zip(self.names, q, strict=True)))
        parts: list[np.ndarray] = []
        for side, hand in mc.HAND_BODIES.items():
            pose = self.probe.body(hand)
            point = pose[:3, :3] @ mc.hand_grip_point(side) + pose[:3, 3]
            parts.append((point - self.grips[side][:3, 3]) / GRIP_WEIGHT_M)
            thumb = pose[:3, :3] @ THUMB_AXIS[side]
            parts.append(np.array([(1.0 - float(thumb @ self.shaft)) / THUMB_WEIGHT]))
        for body, target in self.landmarks.items():
            residual = self.probe.body(body)[:3, 3] - target
            parts.append(residual / landmark_weight(body))
        parts.append(np.asarray(q)[self.rotational] / NEUTRAL_WEIGHT_RAD)
        return np.concatenate(parts)


def free_coordinates(model_xml: mc.Element) -> list[str]:
    return [n for n in mc.unlocked_coordinates(model_xml) if not n.endswith("_beta")]


def _address_range(probe: PoseProbe, name: str) -> tuple[float, float]:
    """``name``'s range in the skeleton, narrowed by :data:`ADDRESS_RANGES`."""
    low, high = probe.coordinate_range(name)
    limit = ADDRESS_RANGES.get(name, (low, high))
    return max(low, limit[0]), min(high, limit[1])


def solve_address(
    skeleton_path: Path, targets: AddressTargets, club: mc.MskClub
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, Any]]:
    """IK of the club-less skeleton; returns ``(q, hand_frames, report)``."""
    from scipy.optimize import least_squares

    from defusedxml import ElementTree as SafeET

    names = free_coordinates(SafeET.parse(str(skeleton_path)).getroot())
    probe = PoseProbe(skeleton_path)
    bounds = np.array([_address_range(probe, n) for n in names])
    grips = grip_frames(club, targets.club_in_ground)
    residual = _Residual(probe, names, targets, grips)
    q0 = np.clip(np.zeros(len(names)), bounds[:, 0], bounds[:, 1])
    pelvis = {"pelvis_tx": 0, "pelvis_ty": 1, "pelvis_tz": 2}
    hips = 0.5 * (targets.landmarks["femur_r"] + targets.landmarks["femur_l"])
    for name, axis in pelvis.items():
        q0[names.index(name)] = hips[axis]
    fit = least_squares(
        residual, q0, bounds=(bounds[:, 0], bounds[:, 1]), x_scale="jac", max_nfev=4000
    )
    q = dict(zip(names, (float(v) for v in fit.x), strict=True))
    probe.set(q)
    hand_frames, gaps = {}, {}
    for side, hand in mc.HAND_BODIES.items():
        pose = probe.body(hand)
        hand_frames[side] = np.linalg.inv(pose) @ grips[side]
        point = pose[:3, :3] @ mc.hand_grip_point(side) + pose[:3, 3]
        gaps[side] = float(np.linalg.norm(point - grips[side][:3, 3]))
    landmark_rms = float(
        np.sqrt(
            np.mean(
                [
                    np.sum((probe.body(b)[:3, 3] - t) ** 2)
                    for b, t in targets.landmarks.items()
                ]
            )
        )
    )
    report = {
        "status": int(fit.status),
        "nfev": int(fit.nfev),
        "hand_grip_point_gap_m": gaps,
        "landmark_rms_m": landmark_rms,
        "floor_native_z_m": targets.floor_native_z,
    }
    return q, hand_frames, report


def skeleton_without_club(model_path: Path, out_dir: Path) -> Path:
    """Write ``model_path`` minus its club to ``out_dir``; returns the new path."""
    from defusedxml import ElementTree as SafeET

    tree = SafeET.parse(str(model_path))
    model = tree.getroot().find("Model")
    if model is None:
        raise ValueError(f"{model_path} has no <Model>")
    mc.strip_club(model)
    out = Path(out_dir) / f"{Path(model_path).stem}_skeleton.osim"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def calibrate(model_path: Path, club: str = "driver") -> mc.GripCalibration:
    """Grip calibration of the Rajagopal model at ``model_path`` for ``club``."""
    shared = mc.load_msk_club(club)
    targets = generated_address_targets(club)
    with tempfile.TemporaryDirectory() as tmp:
        skeleton = skeleton_without_club(Path(model_path), Path(tmp))
        q, frames, report = solve_address(skeleton, targets, shared)
    return mc.GripCalibration(
        model=Path(model_path).stem,
        club=club,
        hand_frames=frames,
        address_q=q,
        club_in_ground=targets.club_in_ground,
        report=report,
    )


def face_angle_deg(model: Any, state: Any, club: mc.MskClub) -> float:
    """Open-positive face angle of the rendered head by OpenSim FK of ``model``.

    The head mesh normal (shared ``clubface_vector``, rolled like the mesh) is
    carried to ground by the ``Club`` body pose and measured in the native
    world with ``club_face.horizontal_face_angle_deg``.
    """
    body = model.getBodySet().get(mc.CLUB_BODY)
    pose = transform_in_ground(body, state)
    normal = pose[:3, :3] @ ca.clubface_vector(club.assembly)
    return cf.horizontal_face_angle_deg(opensim_to_native_vector(normal))


def main(argv: Sequence[str] | None = None) -> int:
    """Calibrate models and store the result in the committed calibration file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", type=Path, help="Rajagopal .osim files")
    parser.add_argument("--club", default="driver", choices=("driver", "iron7"))
    parser.add_argument("--out", type=Path, default=mc.CALIBRATION_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    for path in args.models:
        calibration = calibrate(path, args.club)
        mc.store_calibration(calibration, args.out)
        logger.info("%s %s: %s", calibration.model, args.club, calibration.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
