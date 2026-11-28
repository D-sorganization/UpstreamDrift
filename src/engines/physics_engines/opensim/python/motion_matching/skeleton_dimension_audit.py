"""OpenSim skeleton dimension audit and prescribed kinematic replay gate (A2, #15793)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

Vec3 = tuple[float, float, float]
AxisRecord = tuple[str, Vec3]
GripFrameRecord = tuple[str, Vec3, Vec3]


@dataclass(frozen=True)
class OsimXmlDiagnostic:
    """Structured XML parse failure diagnostic for an OpenSim ``.osim`` file."""

    path: str
    line: int | None
    column: int | None
    message: str
    error_type: str = "ParseError"


@dataclass(frozen=True)
class BodyDimensionRecord:
    """Extracted physical dimensions and attachment metadata for one OpenSim body."""

    name: str
    mass_kg: float
    mass_center_m: Vec3
    inertia_kg_m2: tuple[float, ...]
    parent_joint_name: str | None = None
    parent_body_name: str | None = None
    segment_offset_m: Vec3 = (0.0, 0.0, 0.0)
    segment_length_m: float = 0.0
    joint_axes: tuple[AxisRecord, ...] = ()
    mass_delta_kg: float | None = None


@dataclass(frozen=True)
class OsimSkeletonAuditReport:
    """Fail-closed skeleton dimension audit report for an OpenSim ``.osim`` model."""

    model_path: str
    model_name: str
    passed: bool
    body_count: int
    coordinate_count: int
    muscle_count: int
    total_mass_kg: float
    bodies: tuple[BodyDimensionRecord, ...] = ()
    body_map: dict[str, BodyDimensionRecord] = field(default_factory=dict)
    joint_centers_m: dict[str, Vec3] = field(default_factory=dict)
    joint_axes: dict[str, tuple[AxisRecord, ...]] = field(default_factory=dict)
    grip_frames: dict[str, GripFrameRecord] = field(default_factory=dict)
    mass_mismatches: dict[str, float] = field(default_factory=dict)
    violations: tuple[str, ...] = ()
    diagnostic: OsimXmlDiagnostic | None = None


@dataclass(frozen=True)
class PrescribedReplayTolerances:
    """Acceptance thresholds for prescribed kinematic replay parity gates."""

    landmark_error_m: float = 1e-4
    angular_error_rad: float = 1e-3
    clubhead_error_m: float = 1e-4
    grip_closure_m: float = 1e-3
    capture_a_marker_rms_m: float | None = None


@dataclass(frozen=True)
class PrescribedReplayParityReport:
    """Fail-closed parity gate report for prescribed kinematic replay."""

    passed: bool
    max_landmark_error_m: float
    max_angular_error_rad: float
    max_clubhead_error_m: float
    max_grip_closure_m: float
    capture_a_marker_fit_rms_m: float | None = None
    violations: tuple[str, ...] = ()


def _vec3(text: str | None) -> Vec3:
    """Parse a 3-D vector from XML text."""
    v = [float(t) for t in (text or "").split()]
    if not v:
        return (0.0, 0.0, 0.0)
    if len(v) != 3:
        raise ValueError(f"Expected 3 components, got {len(v)}")
    return (v[0], v[1], v[2])


def _strip_ref(ref: str | None, fallback: str = "") -> str:
    """Normalize an OpenSim socket path to a body or ground name."""
    s = (ref or "").strip()
    if not s or s == "..":
        return fallback
    parts = [p for p in s.split("/") if p and p != "bodyset"]
    return "ground" if s == "/ground" else (parts[0] if parts else s)


def _load_osim(path: Path) -> Any | OsimSkeletonAuditReport:
    """Parse an OpenSim XML file safely, returning a fail-closed report on error."""
    p = str(path)
    if not path.is_file():
        diag = OsimXmlDiagnostic(p, None, None, "File not found", "FileNotFound")
        return OsimSkeletonAuditReport(
            p, "", False, 0, 0, 0, 0.0, violations=("missing_file",), diagnostic=diag
        )
    try:
        root = SafeET.parse(path).getroot()
        model = root if root.tag == "Model" else root.find("Model")
        if model is not None:
            return model
        diag = OsimXmlDiagnostic(p, 1, 1, "Missing <Model> element", "SchemaError")
        viol = "missing_model_element"
    except (SafeET.ParseError, DefusedXmlException, OSError) as exc:
        pos = getattr(exc, "position", None)
        ln, col = (int(pos[0]), int(pos[1])) if pos else (None, None)
        diag = OsimXmlDiagnostic(p, ln, col, str(exc), type(exc).__name__)
        viol = "xml_parse_error"
    return OsimSkeletonAuditReport(
        p, "", False, 0, 0, 0, 0.0, violations=(viol,), diagnostic=diag
    )


def _joint_axes(joint: Any) -> tuple[AxisRecord, ...]:
    """Extract active coordinate axes from a Joint XML element."""
    if t_axes := joint.findall(".//TransformAxis"):
        return tuple(
            (c, _vec3(e.findtext("axis")))
            for e in t_axes
            if (c := (e.findtext("coordinates") or "").strip())
        )
    coords = [c.attrib.get("name", "") for c in joint.findall(".//Coordinate")]
    if joint.tag == "PinJoint" and coords:
        return ((coords[0], (0.0, 0.0, 1.0)),)
    if joint.tag == "UniversalJoint" and len(coords) >= 2:
        return ((coords[0], (1.0, 0.0, 0.0)), (coords[1], (0.0, 1.0, 0.0)))
    return ()


def _collect_frames(
    model: Any,
) -> tuple[dict[str, tuple[str, Vec3]], dict[str, GripFrameRecord]]:
    """Collect PhysicalOffsetFrame entries hosted under Body components."""
    lookup: dict[str, tuple[str, Vec3]] = {}
    grips: dict[str, GripFrameRecord] = {}
    for body in model.findall("./BodySet/objects/Body"):
        bn = body.attrib.get("name", "")
        for f in body.findall("./components/PhysicalOffsetFrame"):
            fn = f.attrib.get("name", "")
            par = _strip_ref(f.findtext("socket_parent"), fallback=bn)
            tr = _vec3(f.findtext("translation"))
            lookup[f"/bodyset/{bn}/{fn}"] = lookup[fn] = (par, tr)
            if "grip" in fn.lower() or "club" in fn.lower():
                grips[fn] = (par, tr, _vec3(f.findtext("orientation")))
    return lookup, grips


def _resolve_joint(
    joint: Any, frames: Mapping[str, tuple[str, Vec3]]
) -> tuple[str | None, str | None, Vec3]:
    """Resolve parent body, child body, and parent offset for a Joint."""
    loc: dict[str, tuple[str, Vec3]] = {}
    for f in joint.findall("./frames/PhysicalOffsetFrame"):
        fn = f.attrib.get("name", "")
        loc[fn] = (
            _strip_ref(f.findtext("socket_parent")),
            _vec3(f.findtext("translation")),
        )
    pr = (joint.findtext("socket_parent_frame") or "").strip()
    cr = (joint.findtext("socket_child_frame") or "").strip()
    pi, ci = loc.get(pr) or frames.get(pr), loc.get(cr) or frames.get(cr)
    return (
        pi[0] if pi else None,
        ci[0] if ci else None,
        pi[1] if pi else (0.0, 0.0, 0.0),
    )


def audit_osim_skeleton_dimensions(
    osim_path: Path | str,
    *,
    reference_masses: Mapping[str, float] | None = None,
    mass_tolerance_kg: float = 1e-3,
) -> OsimSkeletonAuditReport:
    """Audit OpenSim ``.osim`` skeleton geometry, masses, joints, and grip frames."""
    path = Path(osim_path)
    model = _load_osim(path)
    if isinstance(model, OsimSkeletonAuditReport):
        return model

    body_frames, grip_frames = _collect_frames(model)
    joint_centers, joint_axes, child_joints = {}, {}, {}
    for joint in model.findall("./JointSet/objects/*"):
        jn = joint.attrib.get("name", "")
        pb, cb, off = _resolve_joint(joint, body_frames)
        axes = _joint_axes(joint)
        joint_centers[jn], joint_axes[jn] = off, axes
        if cb and cb not in child_joints:
            child_joints[cb] = (jn, pb, off, axes)

    records, mismatches, violations = [], {}, []
    for body in model.findall("./BodySet/objects/Body"):
        bn, mass = body.attrib.get("name", ""), float(body.findtext("mass") or 0.0)
        inert = tuple(
            float(x) for x in (body.findtext("inertia") or "0 0 0 0 0 0").split()
        )
        pj, pb, off, axes = child_joints.get(bn) or (None, None, (0.0, 0.0, 0.0), ())
        delta = (
            mass - float(reference_masses[bn])
            if reference_masses and bn in reference_masses
            else None
        )
        if delta is not None and abs(delta) > mass_tolerance_kg:
            mismatches[bn] = delta
            violations.append(f"mass_mismatch:{bn}")
        com, slen = (
            _vec3(body.findtext("mass_center")),
            math.sqrt(sum(v * v for v in off)),
        )
        records.append(
            BodyDimensionRecord(bn, mass, com, inert, pj, pb, off, slen, axes, delta)
        )

    bmap = {r.name: r for r in records}
    if reference_masses is not None:
        violations.extend(
            f"missing_reference_body:{n}" for n in reference_masses if n not in bmap
        )
    if not records:
        violations.append("empty_bodyset")
    n_c = len(model.findall(".//JointSet/objects//Coordinate"))
    forces = model.findall("./ForceSet/objects/*")
    n_m = sum(1 for f in forces if f.tag.endswith("Muscle"))
    tot, v_tup = sum(r.mass_kg for r in records), tuple(violations)
    return OsimSkeletonAuditReport(
        str(path),
        model.attrib.get("name", ""),
        not v_tup,
        len(records),
        n_c,
        n_m,
        tot,
        tuple(records),
        bmap,
        joint_centers,
        joint_axes,
        grip_frames,
        mismatches,
        v_tup,
    )


def _coerce_pair(
    ref_data: Any, rep_data: Any, label: str
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Coerce paired reference and replay inputs to aligned float64 arrays."""
    empty = np.empty(0, dtype=np.float64)
    if isinstance(ref_data, Mapping) != isinstance(rep_data, Mapping):
        return empty, empty, f"type_mismatch:{label}"
    if isinstance(ref_data, Mapping):
        keys = tuple(sorted(ref_data))
        if keys != tuple(sorted(rep_data)) or not keys:
            return empty, empty, f"key_mismatch:{label}"
        r_a = np.stack([np.asarray(ref_data[k], dtype=float) for k in keys], axis=1)
        p_a = np.stack([np.asarray(rep_data[k], dtype=float) for k in keys], axis=1)
    else:
        r_a = np.asarray(ref_data, dtype=np.float64)
        p_a = np.asarray(rep_data, dtype=np.float64)
    if r_a.shape != p_a.shape or r_a.size == 0:
        return empty, empty, f"shape_mismatch:{label}"
    if not (np.all(np.isfinite(r_a)) and np.all(np.isfinite(p_a))):
        return empty, empty, f"non_finite_values:{label}"
    return r_a, p_a, None


def _capture_a_rms(
    residuals: npt.ArrayLike | float | None,
) -> tuple[float | None, str | None]:
    """Compute scalar capture-A marker-fit RMS separately from replay parity."""
    if residuals is None:
        return None, None
    arr = np.asarray(residuals, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None, "non_finite_values:capture_a_marker_fit"
    if arr.ndim == 0:
        v = float(arr)
        return (v, None) if v >= 0.0 else (None, "invalid_capture_a_rms")
    vals = np.linalg.norm(arr, axis=-1) if arr.ndim >= 2 and arr.shape[-1] == 3 else arr
    return float(np.sqrt(np.mean(vals * vals))), None


def evaluate_prescribed_kinematic_replay(
    reference_landmarks: Mapping[str, npt.ArrayLike] | npt.ArrayLike,
    replay_landmarks: Mapping[str, npt.ArrayLike] | npt.ArrayLike,
    reference_angles: Mapping[str, npt.ArrayLike] | npt.ArrayLike,
    replay_angles: Mapping[str, npt.ArrayLike] | npt.ArrayLike,
    clubhead_pair: tuple[npt.ArrayLike, npt.ArrayLike],
    grip_closure_residuals_m: npt.ArrayLike,
    *,
    capture_a_marker_residuals_m: npt.ArrayLike | float | None = None,
    tolerances: PrescribedReplayTolerances | None = None,
) -> PrescribedReplayParityReport:
    """Evaluate prescribed kinematic replay gates, failing closed on violations."""
    tol = tolerances or PrescribedReplayTolerances()
    ref_lm, rep_lm, e_lm = _coerce_pair(
        reference_landmarks, replay_landmarks, "landmarks"
    )
    ref_ang, rep_ang, e_ang = _coerce_pair(reference_angles, replay_angles, "angles")
    ref_clb, rep_clb, e_clb = _coerce_pair(
        clubhead_pair[0], clubhead_pair[1], "clubhead"
    )
    grip_arr = np.asarray(grip_closure_residuals_m, dtype=np.float64)
    cap_rms, e_cap = _capture_a_rms(capture_a_marker_residuals_m)

    violations = [e for e in (e_lm, e_ang, e_clb, e_cap) if e is not None]
    if grip_arr.size == 0 or not np.all(np.isfinite(grip_arr)):
        violations.append("non_finite_values:grip_closure")
    max_lm = max_ang = max_clb = max_grip = math.nan
    if not violations:
        max_lm = float(np.max(np.linalg.norm(rep_lm - ref_lm, axis=-1)))
        max_ang = float(np.max(np.abs(rep_ang - ref_ang)))
        max_clb = float(np.max(np.linalg.norm(rep_clb - ref_clb, axis=-1)))
        max_grip = float(np.max(np.abs(grip_arr)))
        cap_lim = tol.capture_a_marker_rms_m
        cap_bad = cap_lim is not None and cap_rms is not None and cap_rms > cap_lim
        checks = (
            (max_lm > tol.landmark_error_m, "landmark_error_exceeded"),
            (max_ang > tol.angular_error_rad, "angular_error_exceeded"),
            (max_clb > tol.clubhead_error_m, "clubhead_trajectory_exceeded"),
            (max_grip > tol.grip_closure_m, "grip_closure_exceeded"),
            (cap_bad, "capture_a_marker_rms_exceeded"),
        )
        violations.extend(name for failed, name in checks if failed)
    v_tup = tuple(violations)
    return PrescribedReplayParityReport(
        not v_tup, max_lm, max_ang, max_clb, max_grip, cap_rms, v_tup
    )
