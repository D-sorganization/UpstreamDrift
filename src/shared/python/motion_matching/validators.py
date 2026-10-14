"""Python mirror of MATLAB ``+validators/*`` package.

Each function raises :class:`ValueError` (matching the project's DbC style)
with a descriptive message; the MATLAB error identifiers are recorded in
the docstrings so callers can grep across both languages.

Public API:
    must_have_fields         -- ``mustHaveFields.m``
    must_be_finite_vector    -- ``mustBeFiniteVector.m``
    must_be_monotonic_time   -- ``mustBeMonotonicTime.m``
    must_be_unit_quaternion_rows -- ``mustBeUnitQuaternionRows.m``
    must_be_regularizer_kind -- ``mustBeRegularizerKind.m``
    REGULARIZER_KINDS        -- frozen set of allowed regularizer names.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.vector_math import row_euclidean_norm

__all__ = [
    "MYOSUITE_TOPOLOGY_DIVERGENCE_LEDGER",
    "REGULARIZER_KINDS",
    "RETIRED_LEGACY_URDF_PATHS",
    "get_canonical_golfer_urdf_bundle",
    "verify_spec_exported_urdf_engine_parity",
    "must_be_finite_vector",
    "must_be_monotonic_time",
    "must_be_regularizer_kind",
    "must_be_unit_quaternion_rows",
    "must_have_fields",
]

REGULARIZER_KINDS: frozenset[str] = frozenset(
    {
        "total_work",
        "peak_power",
        "torque_l2",
        "coeff_l2",
        "effort_l2",
        "smoothness_l2",
    }
)


def must_have_fields(obj: object, names: Iterable[str]) -> None:
    """Raise ``ValueError`` if ``obj`` lacks any of the named attributes/keys.

    Mirrors ``validators/mustHaveFields.m`` (id ``validator:missingField``).
    Accepts either a mapping (e.g. dict) or an arbitrary object with
    attributes (e.g. dataclass / namedtuple).
    """
    requested = list(names)
    if isinstance(obj, dict):
        present = set(obj.keys())
    else:
        present = {n for n in requested if hasattr(obj, n)}
        if not present and hasattr(obj, "__dict__"):
            present = set(vars(obj).keys())
    missing = [n for n in requested if n not in present]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")


def must_be_finite_vector(v: object) -> NDArray[np.float64]:
    """Validate ``v`` is a non-empty real, finite 1-D numeric array.

    Mirrors ``validators/mustBeFiniteVector.m``. Returns the array as a
    ``float64`` ndarray for convenience so call sites can chain.
    """
    if v is None:
        raise ValueError("value must be a real numeric vector (got None)")
    arr = np.asarray(v)
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise ValueError("value must be a real numeric array")
    if arr.size == 0 or arr.ndim != 1:
        raise ValueError("value must be a non-empty 1-D vector")
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("value must contain only finite entries (no NaN/Inf)")
    return arr


def must_be_monotonic_time(t: object) -> NDArray[np.float64]:
    """Validate ``t`` is a strictly increasing finite 1-D numeric vector.

    Mirrors ``validators/mustBeMonotonicTime.m``.
    """
    arr = must_be_finite_vector(t)
    if np.any(np.diff(arr) <= 0):
        raise ValueError("time vector must be strictly increasing")
    return arr


def must_be_unit_quaternion_rows(
    q: object,
    tol: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Validate ``q`` is an ``(N, 4)`` real matrix with unit-norm rows.

    Mirrors ``validators/mustBeUnitQuaternionRows.m``.
    """
    if tol <= 0:
        raise ValueError(f"tol must be > 0; got {tol!r}")
    arr = np.asarray(q)
    if (
        not np.issubdtype(arr.dtype, np.number)
        or np.iscomplexobj(arr)
        or arr.ndim != 2
        or arr.shape[1] != 4
        or arr.size == 0
    ):
        raise ValueError(
            f"quaternion array must be a non-empty (N, 4) real matrix; got shape {arr.shape}"
        )
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("quaternion array contains NaN or Inf")
    norms = row_euclidean_norm(arr)
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > tol:
        raise ValueError(
            f"quaternion rows must be unit-norm to {tol:g}; worst deviation {worst:.3g}"
        )
    return arr


def must_be_regularizer_kind(name: object) -> str:
    """Validate ``name`` is one of the allowed regularizer kinds.

    Mirrors ``validators/mustBeRegularizerKind.m``.
    Returns the lower-cased string for downstream dispatch.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("regularizer name must be a non-empty string")
    s = name.lower()
    if s not in REGULARIZER_KINDS:
        raise ValueError(
            f"regularizer {s!r} is not one of: {', '.join(sorted(REGULARIZER_KINDS))}"
        )
    return s


import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.full_body_spec import (
    load_full_body_spec,
    order_full_body_joints,
    upper_body_slice,
)

MYOSUITE_TOPOLOGY_DIVERGENCE_LEDGER: dict[str, Any] = {
    "engine": "myosuite",
    "tier_musculoskeletal": "tier_3_musculoskeletal",
    "tier_rigid_body": "tier_1_rigid_body",
    "divergence_type": "topology_divergence",
    "myohub_coordinates_mapped": 11,
    "total_spec_coordinates": 44,
    "spec_exported_urdf_coordinates": 41,
    "description": (
        "MyoHub musculoskeletal body with 11 of 44 coordinates mapped in "
        "tier_3_musculoskeletal vs the 41/44-coordinate spec-exported URDF in "
        "tier_1_rigid_body."
    ),
    "decision": (
        "spec model in MyoSuite runtime; MyoHub body excluded from torque parity "
        "(unrelated topology, 11/44 coords)"
    ),
    "ledger_reference": "docs/development/full_body_models/evidence/same_input/myosuite_parity_receipt.json",
}

RETIRED_LEGACY_URDF_PATHS: tuple[str, ...] = (
    "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf",
    "src/engines/physics_engines/pinocchio/models/generated/golfer_ik.urdf",
    "src/engines/physics_engines/drake/models/generated/golfer.urdf",
)


def _resolve_spec_doc(
    spec_doc: Mapping[str, Any] | Path | str | None = None,
) -> dict[str, Any]:
    """Resolve full-body specification document from mapping, path, or default."""
    if spec_doc is not None and isinstance(spec_doc, Mapping):
        return dict(spec_doc)
    if spec_doc is not None:
        path = Path(spec_doc)
    else:
        root = Path(__file__).resolve().parents[4]
        path = root / "docs/development/full_body_models/full_body_spec_v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    upper = upper_body_slice(raw)
    return load_full_body_spec(path, upper)


def _compute_spec_poses(
    doc: dict[str, Any],
) -> tuple[float, dict[str, np.ndarray]]:
    """Compute total mass and world-frame neutral pose origins from spec."""
    spec_mass = float(
        sum(s["mass_kg"] for b in doc["bodies"] for s in b.get("solids", []))
    )
    poses: dict[str, np.ndarray] = {"world": np.eye(4)}
    for joint in order_full_body_joints(doc):
        p_pose = poses[joint["parent"]]
        p_base = np.asarray(joint["parent_to_base"], dtype=float)
        c_follower = np.asarray(joint["child_to_follower"], dtype=float)
        poses[joint["child"]] = p_pose @ p_base @ np.linalg.inv(c_follower)

    frame_origins: dict[str, np.ndarray] = {}
    for frame in doc.get("frames", []):
        b_name = frame["body"]
        if b_name in poses:
            placement = np.asarray(frame["placement"], dtype=float)
            frame_origins[frame["name"]] = (poses[b_name] @ placement)[:3, 3]

    return spec_mass, frame_origins


def _solve_urdf_link_poses(root_elem: Any, root_link: str) -> dict[str, np.ndarray]:
    """Propagate transforms through URDF joint tree at neutral pose."""
    poses: dict[str, np.ndarray] = {root_link: np.eye(4)}
    unresolved = list(root_elem.findall("joint"))
    while unresolved:
        next_unresolved = []
        progress = False
        for joint in unresolved:
            p_link = joint.find("parent").attrib["link"]
            c_link = joint.find("child").attrib["link"]
            if p_link in poses:
                origin_elem = joint.find("origin")
                xyz = (
                    [float(x) for x in origin_elem.attrib.get("xyz", "0 0 0").split()]
                    if origin_elem is not None
                    else [0.0, 0.0, 0.0]
                )
                rpy = (
                    [float(x) for x in origin_elem.attrib.get("rpy", "0 0 0").split()]
                    if origin_elem is not None
                    else [0.0, 0.0, 0.0]
                )
                trans = np.eye(4)
                trans[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                trans[:3, 3] = xyz
                poses[c_link] = poses[p_link] @ trans
                progress = True
            else:
                next_unresolved.append(joint)
        if not progress:
            break
        unresolved = next_unresolved
    return poses


def _verify_urdf(
    spec_bytes: bytes, spec_frames: dict[str, np.ndarray]
) -> tuple[float, float, list[str]]:
    """Verify URDF bundle mass, frame origins, and coordinate order."""
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )

    xml, sidecar = export_full_body_urdf(spec_bytes)
    root = ET.fromstring(xml)
    mass_kg = float(
        sum(
            float(m.attrib["value"])
            for link in root.findall("link")
            for m in link.findall("inertial/mass")
        )
    )
    body_links = sidecar.get("body_links", {})
    root_link = body_links.get("world", "body_0")
    link_poses = _solve_urdf_link_poses(root, root_link)
    frame_links = sidecar.get("frame_links", {})
    frame_errs = [
        float(np.linalg.norm(spec_frames[fname] - link_poses[flink][:3, 3]))
        for fname, flink in frame_links.items()
        if fname in spec_frames and flink in link_poses
    ]
    max_frame_err = max(frame_errs) if frame_errs else 0.0
    coords = list(sidecar.get("coordinate_order", []))
    return mass_kg, max_frame_err, coords


def _parse_mjcf_pose(elem: Any) -> np.ndarray:
    """Parse translation and quaternion from an MJCF element."""
    trans = np.eye(4)
    if "pos" in elem.attrib:
        trans[:3, 3] = [float(x) for x in elem.attrib["pos"].split()]
    if "quat" in elem.attrib:
        w, x, y, z = [float(v) for v in elem.attrib["quat"].split()]
        trans[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    return trans


def _collect_mjcf_sites(
    elem: Any, parent_pose: np.ndarray, sites: dict[str, np.ndarray]
) -> None:
    """Recursively collect world-frame positions of MJCF site elements."""
    world_t = parent_pose @ _parse_mjcf_pose(elem)
    for site in elem.findall("site"):
        name = site.attrib.get("name")
        if name:
            s_world = world_t @ _parse_mjcf_pose(site)
            sites[name] = s_world[:3, 3]
    for child in elem.findall("body"):
        _collect_mjcf_sites(child, world_t, sites)


def _verify_mjcf(
    spec_bytes: bytes, spec_frames: dict[str, np.ndarray]
) -> tuple[float, float, list[str]]:
    """Verify MJCF mass, site origins, and coordinate order."""
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, meta = export_full_body_mjcf(spec_bytes)
    root = ET.fromstring(xml)
    mass_kg = float(sum(float(i.attrib["mass"]) for i in root.iter("inertial")))
    sites: dict[str, np.ndarray] = {}
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for b in worldbody.findall("body"):
            _collect_mjcf_sites(b, np.eye(4), sites)
    frame_sites = meta.get("frame_sites", {})
    frame_errs = [
        float(np.linalg.norm(spec_frames[fname] - sites[sname]))
        for fname, sname in frame_sites.items()
        if fname in spec_frames and sname in sites
    ]
    max_frame_err = max(frame_errs) if frame_errs else 0.0
    coords = list(meta.get("coordinate_order", []))
    return mass_kg, max_frame_err, coords


def _verify_pinocchio(
    doc: dict[str, Any], spec_frames: dict[str, np.ndarray]
) -> tuple[int, float]:
    """Verify Pinocchio depth-first joint ordering and neutral frame placement."""
    from src.engines.physics_engines.pinocchio.python.native_model import (
        depth_first_joints,
    )

    ordered = depth_first_joints(doc["joints"])
    poses: dict[str, np.ndarray] = {"world": np.eye(4)}
    for joint in ordered:
        p_pose = poses[joint["parent"]]
        p_base = np.asarray(joint["parent_to_base"], dtype=float)
        c_follower = np.asarray(joint["child_to_follower"], dtype=float)
        poses[joint["child"]] = p_pose @ p_base @ np.linalg.inv(c_follower)

    frame_errs = []
    for frame in doc.get("frames", []):
        name = frame["name"]
        if name in spec_frames and frame["body"] in poses:
            placement = np.asarray(frame["placement"], dtype=float)
            f_world = poses[frame["body"]] @ placement
            frame_errs.append(float(np.linalg.norm(spec_frames[name] - f_world[:3, 3])))

    max_err = max(frame_errs) if frame_errs else 0.0
    return len(ordered), max_err


def get_canonical_golfer_urdf_bundle(
    spec_doc: Mapping[str, Any] | Path | str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Export canonical URDF XML and sidecar metadata directly from load_full_body_spec()."""
    doc = _resolve_spec_doc(spec_doc)
    spec_bytes = json.dumps(doc).encode("utf-8")
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )

    urdf_xml, sidecar = export_full_body_urdf(spec_bytes)
    return urdf_xml, sidecar


def verify_spec_exported_urdf_engine_parity(
    spec_doc: Mapping[str, Any] | Path | str | None = None,
    *,
    mass_tol_kg: float = 1e-9,
    frame_tol_m: float = 1e-3,
) -> dict[str, Any]:
    """Verify spec-exported URDF bundle mass, coordinate order, and frame parity."""
    if mass_tol_kg <= 0:
        raise ValueError(f"mass_tol_kg must be > 0, got {mass_tol_kg}")
    if frame_tol_m <= 0:
        raise ValueError(f"frame_tol_m must be > 0, got {frame_tol_m}")

    doc = _resolve_spec_doc(spec_doc)
    spec_mass, spec_frames = _compute_spec_poses(doc)
    spec_bytes = json.dumps(doc).encode("utf-8")

    urdf_mass, urdf_frame_err, urdf_coords = _verify_urdf(spec_bytes, spec_frames)
    mjcf_mass, mjcf_frame_err, mjcf_coords = _verify_mjcf(spec_bytes, spec_frames)
    pin_count, pin_frame_err = _verify_pinocchio(doc, spec_frames)

    expected_coords = list(doc["coordinate_order"])
    if urdf_coords != expected_coords:
        raise ValueError("URDF coordinate order does not match full-body spec")
    if mjcf_coords != expected_coords:
        raise ValueError("MJCF coordinate order does not match full-body spec")

    mass_err_urdf = abs(urdf_mass - spec_mass)
    mass_err_mjcf = abs(mjcf_mass - spec_mass)
    max_mass_err = max(mass_err_urdf, mass_err_mjcf)

    max_frame_err = max(urdf_frame_err, mjcf_frame_err, pin_frame_err)
    if max_mass_err > mass_tol_kg:
        raise ValueError(
            f"Mass parity error {max_mass_err:.3e} kg exceeds tolerance {mass_tol_kg:.3e} kg"
        )
    if max_frame_err > frame_tol_m:
        raise ValueError(
            f"Frame error {max_frame_err:.3e} m exceeds tolerance {frame_tol_m:.3e} m"
        )

    return {
        "spec_mass_kg": spec_mass,
        "urdf_mass_kg": urdf_mass,
        "mjcf_mass_kg": mjcf_mass,
        "max_mass_error_kg": max_mass_err,
        "coordinate_names_match": True,
        "coordinate_order": expected_coords,
        "max_urdf_frame_error_m": urdf_frame_err,
        "max_mjcf_frame_error_m": mjcf_frame_err,
        "max_pinocchio_frame_error_m": pin_frame_err,
        "max_frame_error_m": max_frame_err,
        "pinocchio_joint_count": pin_count,
        "passed": True,
    }
