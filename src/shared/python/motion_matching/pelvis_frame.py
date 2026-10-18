"""Pelvis frame re-expression and documented hip-centre relocation (HRG-C3).

``hip_calibration.apply_hip_calibration`` moves only the two hip joints. On the
best tour-average driver fit that left the pelvis centre of mass 163 mm and
L5/S1 217 mm above the hip-joint-centre (HJC) midpoint, against 84 mm and
138 mm by design, so the hip moment arms were wrong. This module separates the
repair into two operations that are tested separately.

(a) :func:`reexpress_body_frame` is a pure change of coordinates. It moves the
    origin and axes of one body frame ``F`` to ``F' = F T`` (``T`` the pose of
    the new frame in the old one) and re-expresses everything stored in that
    frame: solid centres of mass and inertia tensors, named frames, child joint
    frames (``parent_to_base``), the joint to the parent (``child_to_follower``),
    contact spheres, closure placements and visual shape centres. The MuJoCo
    body frame (the frame the root coordinates act on) is untouched, so
    coordinates are unchanged and world landmarks, centres of mass, world
    inertia tensors and inverse dynamics are invariant to round-off.

(b) :func:`relocate_pelvis` is a physical change, documented in a
    ``spec_revision`` block and emitted as a NEW document (never an overwrite):

    * the pelvis frame origin is moved to the estimated HJC midpoint (a), axes
      unchanged, i.e. the independent waist-marker plane keeps defining
      orientation (see :func:`pelvic_plane_report`);
    * both hip joints are placed at the supplied centres, hip axes unchanged;
    * the pelvis centre of mass, the L5/S1 pivots (``TorsoInput`` and the spine
      universal joint) and the pelvis visual hint keep their *design offsets
      from the HJC midpoint*, so they move with the midpoint;
    * everything above L5/S1 (trunk bodies, markers, arms, head) keeps its
      fitted pose: the pivots move, the bodies hung on them do not
      (``child_to_follower`` compensates).

Frames: ``Hip`` is the marker (waist) frame, x forward, y left, z up. Units
are metres and kilograms. Software correctness only; scientific qualification
is separate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.full_body_spec import (
    FULL_BODY_SCHEMA_VERSION,
    canonical_sha256,
)

Array: TypeAlias = NDArray[np.float64]
Vec3: TypeAlias = tuple[float, float, float]

PELVIS_BODY = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
HIP_FRAME = "Hip"
HIP_JOINTS = ("hip_r", "hip_l")
REVISION_ISSUE = 13172
ORIENTATION_POLICY = (
    "marker_plane: pelvis axes stay on the waist marker (Hip) frame; the hip "
    "line is not used to orient the frame"
)
_RIGID_TOL = 1.0e-9
_IDENTITY_TOL = 1.0e-12
#: Accepted L5/S1 to HJC-midpoint range (m); the design value is 0.23 x trunk.
L5S1_RANGE_M = (0.13, 0.15)


def _matrix(value: Any, name: str) -> Array:
    out = np.asarray(value, dtype=float)
    if out.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform, got shape {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _check_rigid(transform: Any, name: str) -> Array:
    out = _matrix(transform, name)
    rot = out[:3, :3]
    if (
        float(np.max(np.abs(rot.T @ rot - np.eye(3)))) > _RIGID_TOL
        or abs(float(np.linalg.det(rot)) - 1.0) > _RIGID_TOL
        or float(np.max(np.abs(out[3] - [0.0, 0.0, 0.0, 1.0]))) > _RIGID_TOL
    ):
        raise ValueError(f"{name} must be a proper rigid transform (rotation)")
    return out


def _inverse(transform: Array) -> Array:
    out = np.eye(4)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return out


def _vec3(value: Any, name: str) -> Array:
    out = np.asarray(value, dtype=float)
    if out.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _is_identity_rotation(transform: Array) -> bool:
    return float(np.max(np.abs(transform[:3, :3] - np.eye(3)))) <= _IDENTITY_TOL


def _body_index(spec: Mapping[str, Any], body: str) -> int:
    for k, entry in enumerate(spec["bodies"]):
        if entry["name"] == body:
            return k
    raise ValueError(f"unknown body {body!r}")


# ----------------------------------------------------------------------- (a)


def reexpress_body_frame(
    spec: Mapping[str, Any], body: str, new_in_old: Any
) -> dict[str, Any]:
    """Return a copy of ``spec`` with ``body``'s frame moved to ``F' = F T``.

    ``new_in_old`` is the 4x4 pose ``T`` of the new body frame in the old one.
    Every quantity stored in that frame is re-expressed with ``T^-1``; the
    joint frame the body hangs from does not move, so generalised coordinates
    keep their meaning. Postcondition: for every state the compiled model is
    physically identical (world landmarks, centres of mass, world inertia
    tensors, inverse dynamics), and ``reexpress(T1)`` then ``reexpress(T2)``
    equals ``reexpress(T1 T2)``.

    Raises ``ValueError`` for an unknown body, a non-rigid or non-finite
    ``new_in_old``, or a rotation when the body carries axis-aligned visual
    shape hints (an ellipsoid or box cannot be rotated in the hint schema).
    """
    transform = _check_rigid(new_in_old, "new_in_old")
    index = _body_index(spec, body)
    inv = _inverse(transform)
    rot = inv[:3, :3]
    out = copy.deepcopy(dict(spec))

    solid_names = {s["name"] for s in out["bodies"][index]["solids"]}
    shapes = out.get("visual_hints", {}).get("shapes", {}) or {}
    hinted = sorted(solid_names & set(shapes))
    if hinted and not _is_identity_rotation(transform):
        raise ValueError(
            f"cannot rotate {body!r}: axis-aligned visual shape hints {hinted} "
            "have no orientation (re-express with a pure translation)"
        )

    for solid in out["bodies"][index]["solids"]:
        pose = inv @ np.asarray(solid.get("placement", np.eye(4)), dtype=float)
        com = np.asarray(solid["com_m"], dtype=float)
        tensor = np.asarray(solid["inertia_com_kg_m2"], dtype=float)
        solid["com_m"] = (pose[:3, :3] @ com + pose[:3, 3]).tolist()
        moved = pose[:3, :3] @ tensor @ pose[:3, :3].T
        solid["inertia_com_kg_m2"] = (0.5 * (moved + moved.T)).tolist()
        solid["placement"] = np.eye(4).tolist()
    for name in hinted:
        hint = shapes[name]
        centre = np.asarray(hint["center_m"], dtype=float)
        hint["center_m"] = (rot @ centre + inv[:3, 3]).tolist()
    for frame in out["frames"]:
        if frame["body"] == body:
            frame["placement"] = (inv @ _matrix(frame["placement"], "frame")).tolist()
    for joint in out["joints"]:
        if joint["parent"] == body:
            joint["parent_to_base"] = (
                inv @ _matrix(joint["parent_to_base"], "parent_to_base")
            ).tolist()
        if joint["child"] == body:
            joint["child_to_follower"] = (
                inv @ _matrix(joint["child_to_follower"], "child_to_follower")
            ).tolist()
    for sphere in out["contact"]["spheres"]:
        if sphere["body"] == body:
            point = _vec3(sphere["position_m"], "position_m")
            sphere["position_m"] = (rot @ point + inv[:3, 3]).tolist()
    for suffix in ("a", "b"):
        if out["closure"][f"body_{suffix}"] == body:
            key = f"placement_{suffix}"
            out["closure"][key] = (inv @ _matrix(out["closure"][key], key)).tolist()
    return out


# ----------------------------------------------------------------------- (b)


@dataclass(frozen=True)
class PelvisDesign:
    """Design offsets from the HJC midpoint, in the pelvis frame (metres).

    The anthropometric builder places the pelvis origin at the HJC midpoint,
    the pelvis centre of mass ``com_m`` and the L5/S1 pivot ``l5s1_m`` above it
    (L5/S1 at 0.23 x trunk length, the centre of mass at the de Leva lower
    trunk fraction of that height) and centres the pelvis visual hint at
    ``shape_center_m``. ``apply_hip_calibration`` never moved them, so the
    values stored in a calibrated document are still the design values.
    """

    com_m: Vec3
    l5s1_m: Vec3
    shape_center_m: Vec3 | None

    @classmethod
    def from_spec(
        cls, spec: Mapping[str, Any], pelvis_body: str = PELVIS_BODY
    ) -> PelvisDesign:
        """Read the offsets from ``spec`` (precondition: its pelvis origin is
        the design HJC midpoint, true of every AN-1 and native-derived spec)."""
        entry = spec["bodies"][_body_index(spec, pelvis_body)]
        mass = sum(float(s["mass_kg"]) for s in entry["solids"])
        if mass <= 0.0:
            raise ValueError(f"pelvis body {pelvis_body!r} has no mass")
        com = np.zeros(3)
        for solid in entry["solids"]:
            pose = np.asarray(solid.get("placement", np.eye(4)), dtype=float)
            point = pose[:3, :3] @ np.asarray(solid["com_m"], float) + pose[:3, 3]
            com += float(solid["mass_kg"]) * point / mass
        pivot = _l5s1_pivot(spec, pelvis_body)
        shapes = spec.get("visual_hints", {}).get("shapes", {}) or {}
        centres = [
            shapes[s["name"]]["center_m"]
            for s in entry["solids"]
            if s["name"] in shapes
        ]
        if len(centres) > 1:
            raise ValueError("more than one pelvis visual shape hint")
        return cls(
            tuple(com.tolist()),  # type: ignore[arg-type]
            tuple(pivot.tolist()),  # type: ignore[arg-type]
            tuple(float(c) for c in centres[0]) if centres else None,  # type: ignore[arg-type]
        )


def _l5s1_joints(
    spec: Mapping[str, Any], pelvis_body: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The torso-twist joint off the pelvis and the spine joint above it."""
    twist = [
        j
        for j in spec["joints"]
        if j["parent"] == pelvis_body and j["name"] not in HIP_JOINTS
    ]
    if len(twist) != 1:
        raise ValueError(
            f"expected one L5/S1 twist joint on the pelvis, found {len(twist)}"
        )
    spine = [j for j in spec["joints"] if j["parent"] == twist[0]["child"]]
    if len(spine) != 1:
        raise ValueError(f"expected one spine joint above L5/S1, found {len(spine)}")
    for joint in (twist[0], spine[0]):
        _matrix(joint["parent_to_base"], "parent_to_base")
        _matrix(joint["child_to_follower"], "child_to_follower")
    twist_frame = _follower_pose(twist[0], np.eye(4))
    spine_pivot = (twist_frame @ _matrix(spine[0]["parent_to_base"], "p"))[:3, 3]
    twist_pivot = _matrix(twist[0]["parent_to_base"], "parent_to_base")[:3, 3]
    if float(np.max(np.abs(spine_pivot - twist_pivot))) > _IDENTITY_TOL:
        raise ValueError("the L5/S1 twist and spine joints are not colocated")
    return twist[0], spine[0]


def _follower_pose(joint: Mapping[str, Any], parent_pose: Array) -> Array:
    """Pose of a joint's child body frame in the frame ``parent_pose`` is in."""
    return (
        parent_pose
        @ _matrix(joint["parent_to_base"], "parent_to_base")
        @ _inverse(_matrix(joint["child_to_follower"], "child_to_follower"))
    )


def _shift_pivot(joint: dict[str, Any], parent_pose: Array, delta: Array) -> None:
    """Move ``joint``'s pivot by ``delta`` (frame of ``parent_pose``) in place.

    The child body keeps its pose: ``parent_to_base`` and ``child_to_follower``
    change together. Rotations of the joint frame are preserved.
    """
    moved = np.eye(4)
    moved[:3, 3] = delta
    old = parent_pose @ _matrix(joint["parent_to_base"], "parent_to_base")
    new = moved @ old
    child = _follower_pose(joint, parent_pose)
    joint["parent_to_base"] = (_inverse(parent_pose) @ new).tolist()
    joint["child_to_follower"] = (_inverse(child) @ new).tolist()


def _l5s1_pivot(spec: Mapping[str, Any], pelvis_body: str) -> Array:
    """The L5/S1 pivot in the pelvis frame (twist and spine are colocated)."""
    twist, _ = _l5s1_joints(spec, pelvis_body)
    return _matrix(twist["parent_to_base"], "parent_to_base")[:3, 3]


@dataclass(frozen=True)
class PelvisRevision:
    """A new spec document, its identity, its parent's identity and a report."""

    spec: dict[str, Any]
    canonical_sha256: str
    parent_canonical_sha256: str
    parent_sha256: str
    report: dict[str, Any]


def relocate_pelvis(
    spec: Mapping[str, Any],
    centre_r: Sequence[float],
    centre_l: Sequence[float],
    *,
    pelvis_body: str = PELVIS_BODY,
    hip_frame: str = HIP_FRAME,
    estimate_summary: Mapping[str, Any] | None = None,
) -> PelvisRevision:
    """Documented physical relocation of the hip centres and the pelvis.

    ``centre_r`` and ``centre_l`` are the HJCs in the ``hip_frame`` frame
    (metres). Returns a NEW revision; ``spec`` is not modified. See the module
    docstring for what moves. Postconditions: the hip joints sit exactly at the
    centres; the pelvis centre of mass and L5/S1 pivots are at their design
    offsets from the HJC midpoint; every body above L5/S1, every marker and the
    total mass are unchanged.

    Raises ``ValueError`` for a non-``full-body-v1`` document, missing hip
    joints, a missing ``hip_frame`` on the pelvis, or centres that are not
    finite, 3-vectors and distinct.
    """
    if spec.get("schema_version") != FULL_BODY_SCHEMA_VERSION:
        raise ValueError(f"relocation needs a {FULL_BODY_SCHEMA_VERSION} document")
    c_r, c_l = _vec3(centre_r, "centre_r"), _vec3(centre_l, "centre_l")
    inter = float(np.linalg.norm(c_r - c_l))
    if inter < 1.0e-6:
        raise ValueError("hip centres must be distinct")
    names = {j["name"] for j in spec["joints"]}
    for hip in HIP_JOINTS:
        if hip not in names:
            raise ValueError(f"document has no {hip} joint")
    frames = {f["name"]: f for f in spec["frames"]}
    if hip_frame not in frames or frames[hip_frame]["body"] != pelvis_body:
        raise ValueError(f"frame {hip_frame!r} is not a frame of the pelvis body")
    hip_in_pelvis = _matrix(frames[hip_frame]["placement"], "frame placement")
    design = PelvisDesign.from_spec(spec, pelvis_body)
    twist, spine = _l5s1_joints(spec, pelvis_body)

    def in_pelvis(point: Array) -> Array:
        return hip_in_pelvis[:3, :3] @ point + hip_in_pelvis[:3, 3]

    mid = in_pelvis(0.5 * (c_r + c_l))
    shift = np.eye(4)
    shift[:3, 3] = mid
    out = reexpress_body_frame(spec, pelvis_body, shift)
    # (a) moved everything by -mid; the design offsets are measured from the
    # midpoint, so the pelvis items go back by +mid in the new frame.
    body = out["bodies"][_body_index(out, pelvis_body)]
    for solid in body["solids"]:
        solid["com_m"] = (np.asarray(solid["com_m"], float) + mid).tolist()
    shapes = out.get("visual_hints", {}).get("shapes", {}) or {}
    for solid in body["solids"]:
        if solid["name"] in shapes:
            hint = shapes[solid["name"]]
            hint["center_m"] = (np.asarray(hint["center_m"], float) + mid).tolist()
    by_name = {j["name"]: j for j in out["joints"]}
    new_frame = _matrix(
        {f["name"]: f for f in out["frames"]}[hip_frame]["placement"], "placement"
    )
    for hip, centre in zip(HIP_JOINTS, (c_r, c_l), strict=True):
        placed = _matrix(by_name[hip]["parent_to_base"], "parent_to_base").copy()
        placed[:3, 3] = new_frame[:3, :3] @ centre + new_frame[:3, 3]
        by_name[hip]["parent_to_base"] = placed.tolist()
    # L5/S1: the pivots return to the design offset; the bodies hung on them
    # stay at the fitted pose, so each joint's follower offset takes the shift.
    twist_joint, spine_joint = by_name[twist["name"]], by_name[spine["name"]]
    upper_base = _follower_pose(twist_joint, np.eye(4))
    _shift_pivot(twist_joint, np.eye(4), mid)
    _shift_pivot(spine_joint, upper_base, mid)

    report = _relocation_report(
        spec, out, pelvis_body, hip_frame, c_r, c_l, mid, design
    )
    out["spec_revision"] = {
        "issue": REVISION_ISSUE,
        "name": "hrg-c3-pelvis-relocation",
        "parent_canonical_sha256": canonical_sha256(spec),
        "operations": [
            "reexpress_pelvis_frame_to_hjc_midpoint",
            "relocate_hip_joint_centres",
            "relocate_pelvis_com_l5s1_and_visual_to_design_offsets",
        ],
        "orientation_policy": ORIENTATION_POLICY,
        "hip_frame": hip_frame,
        "hip_centres_hip_frame_m": {"r": c_r.tolist(), "l": c_l.tolist()},
        "hjc_midpoint_pelvis_frame_m": mid.tolist(),
        "design_offsets_from_hjc_midpoint_m": {
            "pelvis_com": list(design.com_m),
            "l5s1": list(design.l5s1_m),
            "pelvis_shape_center": None
            if design.shape_center_m is None
            else list(design.shape_center_m),
        },
        "estimate": dict(estimate_summary) if estimate_summary else None,
        "report": report,
    }
    out["provenance"] = (
        str(spec.get("provenance", ""))
        + " | HRG-C3 (#13172) pelvis revision: pelvis frame origin at the HJC "
        f"midpoint, hip centres relocated (inter-HJC {inter * 1e3:.1f} mm), "
        "pelvis COM / L5-S1 / pelvis visual at design offsets from the "
        "midpoint; upper body pose unchanged; not refitted"
    )
    parent_bytes = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    return PelvisRevision(
        out,
        canonical_sha256(out),
        canonical_sha256(spec),
        hashlib.sha256(parent_bytes.encode("utf-8")).hexdigest(),
        report,
    )


def _relocation_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    pelvis_body: str,
    hip_frame: str,
    c_r: Array,
    c_l: Array,
    mid: Array,
    design: PelvisDesign,
) -> dict[str, Any]:
    """Zero-pose geometry (pelvis frame at the root) before and after."""

    def hjcs(doc: Mapping[str, Any]) -> tuple[Array, Array]:
        joints = {j["name"]: j for j in doc["joints"]}
        return tuple(  # type: ignore[return-value]
            _matrix(joints[h]["parent_to_base"], "parent_to_base")[:3, 3]
            for h in HIP_JOINTS
        )

    def pelvis_com(doc: Mapping[str, Any]) -> Array:
        return np.asarray(PelvisDesign.from_spec(doc, pelvis_body).com_m)

    # All three are expressed in the original pelvis frame so before/after are
    # comparable: the new frame is the old one translated by ``mid``.
    old_r, old_l = hjcs(before)
    new_r, new_l = (h + mid for h in hjcs(after))
    old_mid = 0.5 * (old_r + old_l)
    new_mid = 0.5 * (new_r + new_l)
    old_pivot = _l5s1_pivot(before, pelvis_body)
    new_pivot = _l5s1_pivot(after, pelvis_body) + mid
    old_com = pelvis_com(before)
    new_com = pelvis_com(after) + mid
    l5s1_before = float(np.linalg.norm(old_pivot - old_mid))
    l5s1_after = float(np.linalg.norm(new_pivot - new_mid))
    return {
        "hjc_midpoint_hip_frame_m": (0.5 * (c_r + c_l)).tolist(),
        "inter_hjc_m": float(np.linalg.norm(c_r - c_l)),
        "inter_hjc_before_m": float(np.linalg.norm(old_r - old_l)),
        "hip_centre_displacement_m": {
            "r": (new_r - old_r).tolist(),
            "l": (new_l - old_l).tolist(),
        },
        "pelvis_com_above_hjc_midpoint_before_m": float(
            np.linalg.norm(old_com - old_mid)
        ),
        "pelvis_com_above_hjc_midpoint_after_m": float(
            np.linalg.norm(new_com - new_mid)
        ),
        "l5s1_to_hjc_midpoint_before_m": l5s1_before,
        "l5s1_to_hjc_midpoint_m": l5s1_after,
        "l5s1_within_design_range": bool(
            L5S1_RANGE_M[0] <= l5s1_after <= L5S1_RANGE_M[1]
        ),
        "l5s1_displacement_m": (new_pivot - old_pivot).tolist(),
        "pelvis_com_displacement_m": (new_com - old_com).tolist(),
        "orientation_policy": ORIENTATION_POLICY,
        "design_com_m": list(design.com_m),
        "design_l5s1_m": list(design.l5s1_m),
    }


_ACCEPTED_STATUSES = ("identified",)


def relocate_pelvis_from_estimate(
    spec: Mapping[str, Any],
    estimate: Any,
    *,
    accepted_statuses: Sequence[str] = _ACCEPTED_STATUSES,
    **kwargs: Any,
) -> PelvisRevision:
    """:func:`relocate_pelvis` from a ``HipCentreEstimate`` (HRG-C2).

    Fails closed: an estimate whose status is not in ``accepted_statuses``
    (default only ``identified``) raises ``ValueError``; a prior-dominated
    estimate is a prior, not a measurement, and must be accepted explicitly.
    """
    status = str(estimate.status)
    if status not in accepted_statuses:
        raise ValueError(
            f"hip centre estimate status {status!r} is not accepted "
            f"({', '.join(accepted_statuses)}): "
            f"{getattr(estimate, 'status_reason', '')}"
        )
    summary = {
        "status": status,
        "status_reason": str(getattr(estimate, "status_reason", "")),
        "prior_weight": float(getattr(estimate, "prior_weight", math.nan)),
        "inter_hjc_m": _interval(getattr(estimate, "inter_hjc_m", None)),
        "height_difference_m": _interval(
            getattr(estimate, "height_difference_m", None)
        ),
        "roll_deg": _interval(getattr(estimate, "roll_deg", None)),
    }
    return relocate_pelvis(
        spec,
        estimate.centre_r,
        estimate.centre_l,
        estimate_summary=summary,
        **kwargs,
    )


def _interval(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    return {k: float(getattr(value, k)) for k in ("estimate", "lo", "hi")}


def pelvic_plane_report(estimate: Any) -> dict[str, Any]:
    """Hip-line height difference and roll against the waist marker plane.

    The reference is the plane of the four waist markers, i.e. the ``Hip``
    frame the estimator works in. Aligning a pelvis frame to the hip line would
    make the two HJC heights equal by construction, which is not validation;
    here the hip line is a measured quantity with an interval. If that interval
    contains zero the data do not exclude a level hip line, and the frame is
    NOT rotated to the hip line (``level_hip_line_excluded`` is False).
    Caveat: the same markers also feed the HJC estimate through the pelvis
    frame, so common-mode marker error is shared.
    """
    roll, height = estimate.roll_deg, estimate.height_difference_m
    return {
        "reference": "waist marker plane (Hip frame z axis, +y left)",
        "roll_deg": float(roll.estimate),
        "roll_interval_deg": [float(roll.lo), float(roll.hi)],
        "height_difference_m": float(height.estimate),
        "height_difference_interval_m": [float(height.lo), float(height.hi)],
        "level_hip_line_excluded": bool(roll.lo > 0.0 or roll.hi < 0.0),
        "shared_marker_caveat": "waist markers define both the plane and the estimate",
    }


@dataclass(frozen=True)
class RevisionPaths:
    spec: Path
    receipt: Path


def write_revision(
    revision: PelvisRevision, path: Path, *, receipt_path: Path | None = None
) -> RevisionPaths:
    """Write the spec and its receipt; never overwrite either (FileExistsError).

    The receipt carries the sha256 of the written bytes (``spec_sha256``), the
    canonical hash and the parent's identities. Postcondition: the new
    ``spec_sha256`` differs from the parent's.
    """
    spec_path = Path(path)
    receipt = (
        Path(receipt_path) if receipt_path else spec_path.with_suffix(".receipt.json")
    )
    for target in (spec_path, receipt):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
    text = json.dumps(revision.spec, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest == revision.parent_sha256:
        raise ValueError("revision is byte-identical to its parent")
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(text, encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "issue": REVISION_ISSUE,
                "spec_file": spec_path.name,
                "spec_sha256": digest,
                "spec_canonical_sha256": revision.canonical_sha256,
                "parent_spec_sha256": revision.parent_sha256,
                "parent_spec_canonical_sha256": revision.parent_canonical_sha256,
                "report": revision.report,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return RevisionPaths(spec_path, receipt)
