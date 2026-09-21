"""Full-body model specification: the qualified upper-body spec plus lower limbs
and a contact block (epic #10062, FB-1).

A full-body document embeds the native upper-body geometry document unchanged
(same gravity, coordinates first, bodies, joints, frames and weld closure) and
appends lower-limb bodies and joints, a descriptive contact block, marker
attachments for the tour capture labels, and provenance. Its identity is the
canonical SHA256 of the document. The validator guarantees that the upper-body
slice is byte-identical to the base spec so every engine's full-body variant
stays a strict extension of the qualified model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation
from src.shared.python.motion_matching.tour_capture_contract import tracked_labels


def order_directed_tree(edges: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Require one connected directed tree rooted at world."""
    children = [edge["child"] for edge in edges]
    if len(set(children)) != len(children) or "world" in children:
        raise ValueError("Tree has multiple parents or a cycle through world")
    reached = {"world"}
    pending = list(edges)
    ordered = []
    while pending:
        eligible = [edge for edge in pending if edge["parent"] in reached]
        if not eligible:
            raise ValueError("Joint tree is cyclic or disconnected")
        for edge in eligible:
            ordered.append(edge)
            reached.add(edge["child"])
            pending.remove(edge)
    return ordered


def order_full_body_joints(spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Sequence joints with upper-body tree first, then lower limb chains."""
    upper_spec = upper_body_slice(spec)
    upper_joint_names = {j["name"] for j in upper_spec["joints"]}
    upper_ordered = order_directed_tree(upper_spec["joints"])

    lower_joints_by_name = {
        j["name"]: j for j in spec["joints"] if j["name"] not in upper_joint_names
    }
    leg_chain = [
        "hip_r",
        "knee_r",
        "ankle_r",
        "subtalar_r",
        "mtp_r",
        "hip_l",
        "knee_l",
        "ankle_l",
        "subtalar_l",
        "mtp_l",
    ]
    lower_ordered = [
        lower_joints_by_name[name] for name in leg_chain if name in lower_joints_by_name
    ]
    return list(upper_ordered) + lower_ordered


Array: TypeAlias = NDArray[np.float64]
FULL_BODY_SCHEMA_VERSION = "full-body-v1"
_UPPER_KEYS = (
    "schema_version",
    "qualification",
    "gravity_m_s2",
    "coordinate_order",
    "bodies",
    "joints",
    "frames",
    "closure",
)
_PRIMITIVES = ("Px", "Py", "Pz", "Rx", "Ry", "Rz")
_CONTACT_LAWS = ("hunt_crossley_coulomb",)
_CONTACT_PARAMETERS = (
    "stiffness_n_m",
    "dissipation_s_m",
    "static_friction",
    "dynamic_friction",
    "viscous_friction",
    "transition_velocity_m_s",
)


def canonical_sha256(document: Mapping[str, Any]) -> str:
    """Hash a JSON document independent of key order."""
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite_triple(value: Sequence[float], name: str) -> tuple[float, float, float]:
    v = tuple(float(x) for x in value)
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError(f"{name} must be three finite numbers")
    return v  # type: ignore[return-value]


def _inertia_matrix(value: Sequence[float]) -> list[list[float]]:
    v = [float(x) for x in value]
    if len(v) != 6 or not all(math.isfinite(x) for x in v):
        raise ValueError("Inertia must be six finite numbers (xx, yy, zz, xy, xz, yz)")
    xx, yy, zz, xy, xz, yz = v
    matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    _check_spd(matrix)
    return matrix.tolist()


def _check_spd(matrix: Array) -> None:
    if matrix.shape != (3, 3) or not np.allclose(matrix, matrix.T):
        raise ValueError("Inertia must be a symmetric 3x3 matrix")
    if np.min(np.linalg.eigvalsh(matrix)) <= 0:
        raise ValueError("Inertia must be positive definite")


@dataclass(frozen=True)
class BodySpec:
    """A single-solid rigid body with mass, COM and inertia about the COM."""

    name: str
    mass_kg: float
    com_m: tuple[float, float, float]
    inertia_com_kg_m2: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Body name required")
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("Body mass must be positive")
        object.__setattr__(self, "com_m", _finite_triple(self.com_m, "com_m"))
        _inertia_matrix(self.inertia_com_kg_m2)

    def as_document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "solids": [
                {
                    "name": self.name,
                    "mass_kg": self.mass_kg,
                    "com_m": list(self.com_m),
                    "inertia_com_kg_m2": _inertia_matrix(self.inertia_com_kg_m2),
                    "placement": np.eye(4).tolist(),
                }
            ],
        }


def _transform(value: Any, name: str) -> list[list[float]]:
    m = np.asarray(value, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(m[3], [0, 0, 0, 1]) or not np.allclose(
        m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9
    ):
        raise ValueError(f"{name} must be a rigid homogeneous transform")
    return m.tolist()


@dataclass(frozen=True)
class JointSpec:
    """A joint edge with native primitive sequence and coordinate names."""

    name: str
    parent: str
    child: str
    parent_to_base: list[list[float]]
    child_to_follower: list[list[float]]
    primitives: tuple[str, ...]
    coordinates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.parent.strip() or not self.child.strip():
            raise ValueError("Joint name, parent and child are required")
        if not self.primitives or any(p not in _PRIMITIVES for p in self.primitives):
            raise ValueError("Joint primitives must be from Px, Py, Pz, Rx, Ry, Rz")
        if len(self.coordinates) != len(self.primitives) or len(
            set(self.coordinates)
        ) != len(self.coordinates):
            raise ValueError("One unique coordinate per primitive is required")
        object.__setattr__(
            self, "parent_to_base", _transform(self.parent_to_base, "parent_to_base")
        )
        object.__setattr__(
            self,
            "child_to_follower",
            _transform(self.child_to_follower, "child_to_follower"),
        )

    def as_document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "child": self.child,
            "parent_to_base": self.parent_to_base,
            "child_to_follower": self.child_to_follower,
            "primitives": [
                {"primitive": p, "coordinate": c}
                for p, c in zip(self.primitives, self.coordinates, strict=True)
            ],
        }


@dataclass(frozen=True)
class LowerLimbExtension:
    bodies: Sequence[BodySpec]
    joints: Sequence[JointSpec]
    provenance: str

    def __post_init__(self) -> None:
        if not self.bodies or not self.joints or not self.provenance.strip():
            raise ValueError("Extension needs bodies, joints and provenance")


@dataclass(frozen=True)
class ContactSphere:
    name: str
    body: str
    position_m: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.body.strip():
            raise ValueError("Contact sphere name and body required")
        object.__setattr__(
            self, "position_m", _finite_triple(self.position_m, "position_m")
        )
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("Contact sphere radius must be positive")


@dataclass(frozen=True)
class ContactSpec:
    law: str
    parameters: Mapping[str, float]
    spheres: Sequence[ContactSphere]
    ground_normal_policy: str
    ground_height_m: float | None
    provenance: str

    def __post_init__(self) -> None:
        if self.law not in _CONTACT_LAWS:
            raise ValueError(f"Unsupported contact law {self.law}")
        if set(self.parameters) != set(_CONTACT_PARAMETERS) or not all(
            math.isfinite(float(v)) for v in self.parameters.values()
        ):
            raise ValueError("Contact parameters must be exactly the shared law set")
        if not self.spheres or self.ground_normal_policy != "opposite_gravity":
            raise ValueError(
                "Contact needs spheres and the opposite_gravity ground policy"
            )
        if self.ground_height_m is not None and not math.isfinite(self.ground_height_m):
            raise ValueError("Ground height must be finite or None (uncalibrated)")
        if not self.provenance.strip():
            raise ValueError("Contact provenance required")

    def as_document(self) -> dict[str, Any]:
        return {
            "law": self.law,
            "parameters": {k: float(self.parameters[k]) for k in _CONTACT_PARAMETERS},
            "spheres": [
                {
                    "name": s.name,
                    "body": s.body,
                    "position_m": list(s.position_m),
                    "radius_m": s.radius_m,
                }
                for s in self.spheres
            ],
            "ground": {
                "normal_policy": self.ground_normal_policy,
                "height_m": self.ground_height_m,
                "calibrated": self.ground_height_m is not None,
            },
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class MarkerAttachment:
    body: str
    offset_m: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        if not self.body.strip():
            raise ValueError("Marker body required")
        if self.offset_m is not None:
            object.__setattr__(
                self, "offset_m", _finite_triple(self.offset_m, "offset_m")
            )


def derive_full_body_spec(
    upper: Mapping[str, Any],
    extension: LowerLimbExtension,
    contact: ContactSpec,
    markers: Mapping[str, MarkerAttachment],
    *,
    provenance: str,
) -> dict[str, Any]:
    """Compose and validate a full-body document from its parts."""
    base = {key: json.loads(json.dumps(upper[key])) for key in _UPPER_KEYS}
    document = {
        **base,
        "schema_version": FULL_BODY_SCHEMA_VERSION,
        "upper_body_schema_version": upper["schema_version"],
        "upper_body_qualification": upper["qualification"],
        "upper_body_sha256": canonical_sha256(upper),
        "upper_body_counts": {
            "bodies": len(upper["bodies"]),
            "joints": len(upper["joints"]),
            "coordinates": len(upper["coordinate_order"]),
        },
        "coordinate_order": list(upper["coordinate_order"])
        + [c for joint in extension.joints for c in joint.coordinates],
        "bodies": list(base["bodies"]) + [b.as_document() for b in extension.bodies],
        "joints": list(base["joints"]) + [j.as_document() for j in extension.joints],
        "lower_limb_provenance": extension.provenance,
        "contact": contact.as_document(),
        "marker_attachments": {
            label: {
                "body": m.body,
                "offset_m": None if m.offset_m is None else list(m.offset_m),
            }
            for label, m in markers.items()
        },
        "provenance": provenance,
        "qualification": "full-body extension of native-derived geometry; dynamics unqualified",
    }
    validate_full_body_spec(document, upper)
    return document


def upper_body_slice(document: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the embedded upper-body document from its recorded counts."""
    counts = document["upper_body_counts"]
    n_bodies, n_joints, n_coordinates = (
        int(counts["bodies"]),
        int(counts["joints"]),
        int(counts["coordinates"]),
    )
    return {
        "schema_version": document["upper_body_schema_version"],
        "qualification": document["upper_body_qualification"],
        "gravity_m_s2": document["gravity_m_s2"],
        "coordinate_order": list(document["coordinate_order"][:n_coordinates]),
        "bodies": document["bodies"][:n_bodies],
        "joints": document["joints"][:n_joints],
        "frames": document["frames"],
        "closure": document["closure"],
    }


def validate_full_body_spec(
    document: Mapping[str, Any], upper: Mapping[str, Any]
) -> None:
    """Raise ValueError unless ``document`` is a valid strict extension of ``upper``."""
    if document.get("schema_version") != FULL_BODY_SCHEMA_VERSION:
        raise ValueError("Not a full-body-v1 document")
    if document.get("upper_body_sha256") != canonical_sha256(upper):
        raise ValueError("Embedded upper-body identity differs from the supplied base")
    counts = document.get("upper_body_counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "bodies",
        "joints",
        "coordinates",
    }:
        raise ValueError("Missing upper-body counts")
    if upper_body_slice(document) != {key: upper[key] for key in _UPPER_KEYS}:
        raise ValueError("Upper-body slice is not byte-identical to the base spec")
    gravity = np.asarray(document["gravity_m_s2"], dtype=float)
    if (
        gravity.shape != (3,)
        or not np.isfinite(gravity).all()
        or np.linalg.norm(gravity) <= 0
    ):
        raise ValueError("Gravity must be a finite nonzero vector")
    coordinates = list(document["coordinate_order"])
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Duplicate coordinate names")
    joint_coordinates = [
        p["coordinate"] for j in document["joints"] for p in j["primitives"]
    ]
    if sorted(joint_coordinates) != sorted(coordinates):
        raise ValueError("Coordinate order and joint primitives disagree")
    bodies = {b["name"] for b in document["bodies"]}
    if len(bodies) != len(document["bodies"]):
        raise ValueError("Duplicate body names")
    for body in document["bodies"]:
        for solid in body["solids"]:
            if not math.isfinite(solid["mass_kg"]) or solid["mass_kg"] < 0:
                raise ValueError(f"Negative or nonfinite mass on {solid['name']}")
            if solid["mass_kg"] > 0:
                _check_spd(np.asarray(solid["inertia_com_kg_m2"], dtype=float))
    for joint in document["joints"]:
        if joint["parent"] != "world" and joint["parent"] not in bodies:
            raise ValueError(f"Joint {joint['name']} parent is not a body")
        if joint["child"] not in bodies:
            raise ValueError(f"Joint {joint['name']} child is not a body")
    ordered = order_directed_tree(document["joints"])
    if {"world"} | {j["child"] for j in ordered} != bodies | {"world"}:
        raise ValueError(
            "Every body must be reachable from world through exactly one joint"
        )
    contact = document["contact"]
    ContactSpec(
        contact["law"],
        contact["parameters"],
        tuple(
            ContactSphere(s["name"], s["body"], tuple(s["position_m"]), s["radius_m"])
            for s in contact["spheres"]
        ),
        contact["ground"]["normal_policy"],
        contact["ground"]["height_m"],
        contact["provenance"],
    )
    for sphere in contact["spheres"]:
        if sphere["body"] not in bodies:
            raise ValueError(f"Contact sphere {sphere['name']} references unknown body")
    allowed = set(tracked_labels())
    for label, attachment in document["marker_attachments"].items():
        if label not in allowed:
            raise ValueError(f"Unknown capture label {label}")
        if attachment["body"] not in bodies and attachment["body"] not in {
            f["name"] for f in document["frames"]
        }:
            raise ValueError(f"Marker {label} attached to unknown body or frame")
        MarkerAttachment(
            attachment["body"],
            None if attachment["offset_m"] is None else tuple(attachment["offset_m"]),
        )
    if (
        not str(document.get("provenance", "")).strip()
        or not str(document.get("lower_limb_provenance", "")).strip()
    ):
        raise ValueError("Provenance is required")


def save_full_body_spec(document: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_full_body_spec(path: Path, upper: Mapping[str, Any]) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_full_body_spec(document, upper)
    return document


def pelvis_alignment(
    target_offsets: Mapping[str, Sequence[float]],
    source_offsets: Mapping[str, Sequence[float]],
) -> tuple[Array, float]:
    """Rigid transform mapping source-frame points onto target-frame points.

    Both mappings share at least three labels. Returns (4x4 transform, RMS
    residual in metres) with ``target ≈ transform @ source`` for each label.
    """
    labels = [label for label in target_offsets if label in source_offsets]
    if len(labels) < 3:
        raise ValueError("Pelvis alignment needs at least three shared markers")
    target = np.asarray([target_offsets[label] for label in labels], dtype=float)
    source = np.asarray([source_offsets[label] for label in labels], dtype=float)
    if (
        target.shape != (len(labels), 3)
        or not np.isfinite(target).all()
        or not np.isfinite(source).all()
    ):
        raise ValueError("Marker offsets must be finite 3-vectors")
    tc, sc = target.mean(axis=0), source.mean(axis=0)
    rotation = kabsch_rotation(source - sc, target - tc)
    translation = tc - rotation @ sc
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    diff = source @ rotation.T + translation - target
    residual = float(
        np.sqrt(
            np.mean(
                np.einsum("...i,...i->...", diff, diff)
            )  # ⚡ Bolt: np.einsum is ~2x faster than np.sum(diff ** 2, axis=1)
        )
    )
    return transform, residual
