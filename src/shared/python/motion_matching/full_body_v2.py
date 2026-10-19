"""The ``full-body-v2`` schema: two regional trunk joints (HRG-D1, #13170).

A v2 document is a v1 anthropometric document with the single-pivot trunk
(``TorsoInput`` then ``SpineInputX/Y``, all at L5/S1, above a 0.3 kg dummy
body) replaced by two regional lumped 3-DOF joints. This module is the one
declarative place for the v2 trunk; HRG-D2 (#13173) builds its dynamics.

Topology (pelvis frame at q = 0: x forward, y left, z up)::

    pelvis -[lumbar_L5_S1: Ry, Rx, Rz]-> trunk_lumbar  (de Leva middle_trunk)
           -[thoracic_T12_L1: Ry, Rx, Rz]-> thorax      (v1 COMRod: upper_trunk,
                                                         neck, scapulae, Hub)

The thorax keeps the v1 ``COMRod`` body name, so every joint, frame, marker
and consumer above the trunk is unchanged. The dummy ``UpperTorsoBase`` is
dropped (recorded with its mass); its frames move to the lumbar body with
their neutral pose preserved.

Coordinates are model-axis primitive angles in the sequence Ry, Rx, Rz, which
is the ISB ``isb_spine`` ZXY sequence through ``MODEL_TO_ISB``
(ADR-0055 item 1): ``FE = -flexion``, ``LB = +lateral_bending`` (right),
``AR = +axial_rotation`` (left). Soft limits are declared in ISB sign
(extension positive) as HRG-D2 specifies; ``soft_limits_deg`` carries them
in model sign (never ``coordinate_ranges_deg``, which IK treats as hard
bounds). Joint locations are an explicit input (:class:`TrunkJointLocations`);
:func:`segment_boundary_locations` is the declared HRG-D1 stand-in and
``full_body_v2_geometry.landmark_trunk_locations`` the HRG-D2 landmark one.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.topology_contact import (
    HAND_HAND_IDENTIFIABILITY_NOTE,
)
from src.shared.python.motion_matching.full_body_schema import (
    FullBodySchema,
    TrunkJointDefinition,
    TrunkTopology,
    require_full_body_schema,
)
from src.shared.python.motion_matching.full_body_spec import (
    V1_TRUNK,
    JointSpec,
    canonical_sha256,
    order_directed_tree,
    upper_body_slice,
    validate_full_body_structure,
    validate_upper_body_identity,
)

Array = NDArray[np.float64]

__all__ = [
    "CLUB_FREE_COORDINATES",
    "CLUB_FREE_JOINT",
    "FULL_BODY_SCHEMA",
    "FULL_BODY_V2_SCHEMA_VERSION",
    "HARD_RANGE_POLICY",
    "LEFT_HAND_BODY",
    "LUMBAR_BODY",
    "V2_TRUNK",
    "V2_UPPER_BODY_SCHEMA_VERSION",
    "TrunkJointLocations",
    "derive_full_body_v2_spec",
    "segment_boundary_locations",
    "separate_v2_hands",
    "trunk_bodies",
    "v2_coordinate_order",
    "v2_trunk_ranges_deg",
    "validate_full_body_v2_spec",
]

FULL_BODY_V2_SCHEMA_VERSION = "full-body-v2"
V2_UPPER_BODY_SCHEMA_VERSION = "full-body-v2-upper"
LUMBAR_BODY = "trunk_lumbar"
LUMBAR_SOLID = "middle_trunk"
THORAX_SOLID = "upper_trunk"
LEFT_HAND_BODY = "solid_reference:GolfSwing3D_Kinetic/Grip/LHandBody"
CLUB_FREE_JOINT = "GolfSwing3D_Kinetic/Grip/ClubFreeJoint"
CLUB_FREE_COORDINATES = (
    "ClubFreeRx",
    "ClubFreeRy",
    "ClubFreeRz",
    "ClubFreePx",
    "ClubFreePy",
    "ClubFreePz",
)
_FREE_PRIMITIVES = ("Rx", "Ry", "Rz", "Px", "Py", "Pz")
_LEFT_HAND_SOLIDS = ("LHand", "LHandStandoff")
_RIGHT_HAND_SOLIDS = ("RHand",)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REGIONAL_PRIMITIVES = ("Ry", "Rx", "Rz")
_ISB_ORDER = ("flexion_extension", "lateral_bending", "axial_rotation")
_ISB_SIGNS = (-1.0, 1.0, 1.0)
#: HRG-D2: the trunk ROM is a soft limit (``soft_limits_deg``, model sign).
#: It is never written to ``coordinate_ranges_deg``, which IK consumers treat
#: as hard bounds; an exceedance is reported, never clamped.
HARD_RANGE_POLICY = (
    "report_only: trunk ROM is soft (soft_limits_deg); exceedances are "
    "reported and never enforced as IK or engine limits"
)


def _regional(
    name: str, level: str, roles: tuple[str, str], prefix: str, limits: tuple
) -> TrunkJointDefinition:
    return TrunkJointDefinition(
        name=name,
        level=level,
        parent_role=roles[0],
        child_role=roles[1],
        primitives=_REGIONAL_PRIMITIVES,
        coordinates=(
            f"{prefix}_flexion",
            f"{prefix}_lateral_bending",
            f"{prefix}_axial_rotation",
        ),
        isb_axes=_ISB_ORDER,
        isb_signs=_ISB_SIGNS,
        soft_limits_isb_deg=limits,
    )


#: The v2 trunk. ROM (ISB sign, degrees) from HRG-D2 #13173: lumbar FE
#: -40..+15, LB +-20, AR +-15; thoracic FE -30..+15, LB +-25, AR +-50.
V2_TRUNK = TrunkTopology(
    joints=(
        _regional(
            "lumbar_L5_S1",
            "L5_S1",
            ("pelvis", "lumbar"),
            "lumbar",
            ((-40.0, 15.0), (-20.0, 20.0), (-15.0, 15.0)),
        ),
        _regional(
            "thoracic_T12_L1",
            "T12_L1",
            ("lumbar", "thorax"),
            "thoracic",
            ((-30.0, 15.0), (-25.0, 25.0), (-50.0, 50.0)),
        ),
    ),
    coordinates=(
        "lumbar_flexion",
        "lumbar_lateral_bending",
        "lumbar_axial_rotation",
        "thoracic_flexion",
        "thoracic_lateral_bending",
        "thoracic_axial_rotation",
    ),
    description=(
        "regional lumped 3-DOF joints labelled at L5/S1 (lumbar) and T12/L1 "
        "(thoracic), Ry-Rx-Rz about model axes = isb_spine ZXY"
    ),
)


def v2_trunk_ranges_deg() -> dict[str, tuple[float, float]]:
    """Soft limits of the v2 trunk coordinates in model sign (degrees).

    Postcondition: one ordered ``(lo, hi)`` per v2 trunk coordinate; a
    coordinate with ISB sign -1 maps ISB ``(lo, hi)`` to ``(-hi, -lo)``.
    """
    out: dict[str, tuple[float, float]] = {}
    for joint in V2_TRUNK.joints:
        limits = joint.soft_limits_isb_deg or ()
        for name, sign, (lo, hi) in zip(
            joint.coordinates, joint.isb_signs, limits, strict=True
        ):
            pair = (float(lo), float(hi)) if sign > 0 else (-float(hi), -float(lo))
            out[name] = pair
    return out


def v2_coordinate_order(v1_order: Sequence[str]) -> list[str]:
    """Replace the v1 trunk coordinates with the v2 ones, at the v1 position.

    Raises ``ValueError`` unless every v1 trunk coordinate is present once.
    Postcondition: every other coordinate keeps its relative order.
    """
    order = list(v1_order)
    missing = [c for c in V1_TRUNK.coordinates if order.count(c) != 1]
    if missing:
        raise ValueError(f"v1 coordinate order lacks trunk coordinates {missing}")
    first = min(order.index(c) for c in V1_TRUNK.coordinates)
    rest = [c for c in order if c not in V1_TRUNK.coordinates]
    return rest[:first] + list(V2_TRUNK.coordinates) + rest[first:]


def _vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    out = tuple(float(x) for x in value)
    if len(out) != 3 or not all(math.isfinite(x) for x in out):
        raise ValueError(f"{name} must be three finite numbers")
    return out  # type: ignore[return-value]


@dataclass(frozen=True)
class TrunkJointLocations:
    """Neutral (q = 0) joint centres in the pelvis frame, metres.

    Preconditions (``ValueError``): finite 3-vectors, at least 1 mm apart,
    non-empty provenance saying where the locations came from; ``evidence``
    (optional, JSON-serialisable mapping) is recorded verbatim in the
    document's ``trunk_topology.locations_evidence``.
    """

    l5s1_in_pelvis_m: tuple[float, float, float]
    t12l1_in_pelvis_m: tuple[float, float, float]
    provenance: str
    evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        lo = _vec3(self.l5s1_in_pelvis_m, "l5s1_in_pelvis_m")
        up = _vec3(self.t12l1_in_pelvis_m, "t12l1_in_pelvis_m")
        if math.dist(lo, up) < 1e-3:
            raise ValueError("L5/S1 and T12/L1 must be distinct (>= 1 mm apart)")
        if not self.provenance.strip():
            raise ValueError("Trunk joint locations need a provenance")
        if self.evidence is not None and not isinstance(self.evidence, Mapping):
            raise TypeError("Trunk joint location evidence must be a Mapping")
        object.__setattr__(self, "l5s1_in_pelvis_m", lo)
        object.__setattr__(self, "t12l1_in_pelvis_m", up)


def _translation(vector: ArrayLike) -> Array:
    out = np.eye(4)
    out[:3, 3] = np.asarray(vector, dtype=float)
    return out


def _joint_pose_at_zero(joint: Mapping[str, Any]) -> Array:
    """Child pose in the parent at q = 0: ``parent_to_base @ inv(child_to_follower)``."""
    base = np.asarray(joint["parent_to_base"], dtype=float)
    return base @ np.linalg.inv(np.asarray(joint["child_to_follower"], dtype=float))


def _pelvis(spec: Mapping[str, Any]) -> str:
    roots = [
        j["child"]
        for j in spec["joints"]
        if j["parent"] == "world" and j["name"] != CLUB_FREE_JOINT
    ]
    if len(roots) != 1:
        raise ValueError(f"Expected one world root joint, found {len(roots)}")
    return roots[0]


def _neutral_body_poses(spec: Mapping[str, Any]) -> dict[str, Array]:
    """World 4x4 poses of all bodies in ``spec`` at q = 0."""
    poses: dict[str, Array] = {"world": np.eye(4)}
    for joint in order_directed_tree(spec["joints"]):
        poses[joint["child"]] = poses[joint["parent"]] @ _joint_pose_at_zero(joint)
    return poses


def trunk_bodies(spec: Mapping[str, Any]) -> dict[str, str]:
    """Body names by trunk role (``pelvis``, intermediate role, ``thorax``).

    Uses the declared trunk of the spec's own schema. Raises ``ValueError``
    when the trunk is missing or not rooted at the pelvis.
    """
    trunk = require_full_body_schema(spec, consumer="trunk_bodies").trunk
    joints = trunk.locate(spec)
    pelvis = _pelvis(spec)
    if joints[0]["parent"] != pelvis:
        raise ValueError("The trunk must start at the pelvis (world root child)")
    roles = {"pelvis": pelvis}
    for joint_def, joint in zip(trunk.joints, joints, strict=True):
        roles[joint_def.child_role] = joint["child"]
    return roles


def _solid(body: Mapping[str, Any], suffix: str) -> Mapping[str, Any]:
    matches = [s for s in body["solids"] if s["name"].rsplit("/", 1)[-1] == suffix]
    if len(matches) != 1:
        raise ValueError(
            f"v2 derivation needs one de Leva {suffix} solid on {body['name']!r} "
            "(anthropometric family); native specs carry none"
        )
    return matches[0]


def _v1_trunk_geometry(v1: Mapping[str, Any]) -> dict[str, Any]:
    """Bodies, joints and neutral pelvis-frame poses of the v1 trunk."""
    roles = trunk_bodies(v1)
    twist, spine = V1_TRUNK.locate(v1)
    bodies = {b["name"]: b for b in v1["bodies"]}
    thorax = bodies[roles["thorax"]]
    _solid(thorax, LUMBAR_SOLID)
    _solid(thorax, THORAX_SOLID)
    dummy_pose = _joint_pose_at_zero(twist)
    return {
        "roles": roles,
        "twist": twist,
        "spine": spine,
        "dummy_pose": dummy_pose,
        "thorax_pose": dummy_pose @ _joint_pose_at_zero(spine),
    }


def segment_boundary_locations(v1: Mapping[str, Any]) -> TrunkJointLocations:
    """HRG-D1 stand-in joint centres from the v1 anthropometric geometry.

    L5/S1 is the v1 twist pivot. T12/L1 is the xiphion boundary between the
    de Leva middle and upper trunk solids, which the anthropometric builder
    places halfway from L5/S1 to the shoulder ``Hub`` frame. These are not
    landmark estimates; HRG-D2 replaces them (and checks lordosis/kyphosis).

    Raises ``ValueError`` for a non-v1 document, a native spec without the
    de Leva solids, or a missing ``Hub`` frame on the thorax.
    """
    require_full_body_schema(
        v1, consumer="segment_boundary_locations", accepted=("full-body-v1",)
    )
    geometry = _v1_trunk_geometry(v1)
    hubs = [
        f
        for f in v1["frames"]
        if f["name"] == "Hub" and f["body"] == geometry["roles"]["thorax"]
    ]
    if len(hubs) != 1:
        raise ValueError("segment_boundary_locations needs the thorax Hub frame")
    hub = np.asarray(hubs[0]["placement"], dtype=float)[:3, 3]
    thorax = geometry["thorax_pose"]
    pivot = geometry["twist"]["parent_to_base"]
    return TrunkJointLocations(
        tuple(np.asarray(pivot, dtype=float)[:3, 3]),
        tuple((thorax @ np.append(0.5 * hub, 1.0))[:3]),
        "HRG-D1 stand-in: L5/S1 at the v1 twist pivot; T12/L1 at the xiphion "
        "boundary of the de Leva middle/upper trunk solids (midway to the Hub). "
        "Not landmark-derived; HRG-D2 replaces it.",
    )


def _regional_joint(
    joint_def: TrunkJointDefinition,
    bodies: tuple[str, str],
    placement: tuple[Array, Array],
) -> dict[str, Any]:
    return JointSpec(
        joint_def.name,
        bodies[0],
        bodies[1],
        placement[0].tolist(),
        placement[1].tolist(),
        joint_def.primitives,
        joint_def.coordinates,
    ).as_document()


def _replace_trunk(
    v1: Mapping[str, Any], locations: TrunkJointLocations
) -> dict[str, Any]:
    """Bodies, joints and frames of the v2 document plus a trunk record."""
    geo = _v1_trunk_geometry(v1)
    roles = geo["roles"]
    dummy, thorax = roles["twist_base"], roles["thorax"]
    lumbar0 = _translation(locations.l5s1_in_pelvis_m)
    base_t = _translation(
        np.subtract(locations.t12l1_in_pelvis_m, locations.l5s1_in_pelvis_m)
    )
    to_lumbar = np.linalg.inv(lumbar0)
    lumbar_joint, thoracic_joint = V2_TRUNK.joints
    joints = {
        geo["twist"]["name"]: _regional_joint(
            lumbar_joint, (roles["pelvis"], LUMBAR_BODY), (lumbar0, np.eye(4))
        ),
        geo["spine"]["name"]: _regional_joint(
            thoracic_joint,
            (LUMBAR_BODY, thorax),
            (base_t, np.linalg.inv(geo["thorax_pose"]) @ lumbar0 @ base_t),
        ),
    }
    copied = json.loads(json.dumps(v1))
    thorax_body = next(b for b in copied["bodies"] if b["name"] == thorax)
    moved = dict(_solid(thorax_body, LUMBAR_SOLID))
    thorax_body["solids"] = [
        s for s in thorax_body["solids"] if s["name"] != moved["name"]
    ]
    placement = to_lumbar @ geo["thorax_pose"] @ np.asarray(moved["placement"])
    moved["placement"] = placement.tolist()
    lumbar_body = {"name": LUMBAR_BODY, "solids": [moved]}
    dropped = next(b for b in copied["bodies"] if b["name"] == dummy)
    bodies = [lumbar_body if b["name"] == dummy else b for b in copied["bodies"]]
    frames = []
    for frame in copied["frames"]:
        if frame["body"] == dummy:
            pose = to_lumbar @ geo["dummy_pose"] @ np.asarray(frame["placement"])
            frame = {**frame, "body": LUMBAR_BODY, "placement": pose.tolist()}
        frames.append(frame)
    record = {
        "dropped_bodies": [
            {"body": dummy, "mass_kg": sum(s["mass_kg"] for s in dropped["solids"])}
        ],
        "reparented_solids": [
            {"solid": moved["name"], "from": thorax, "to": LUMBAR_BODY}
        ],
    }
    return {
        "bodies": bodies,
        "joints": [joints.get(j["name"], j) for j in copied["joints"]],
        "frames": frames,
        "record": record,
    }


def _check_dummy_is_free(v1: Mapping[str, Any], dummy: str) -> None:
    closure = v1["closure"]
    used = {closure.get("body_a"), closure.get("body_b")}
    used |= {s["body"] for s in v1["contact"]["spheres"]}
    used |= {m["body"] for m in v1["marker_attachments"].values()}
    if dummy in used:
        raise ValueError(f"v2 derivation cannot drop {dummy!r}: it is referenced")


def _trunk_record(
    locations: TrunkJointLocations, replaced: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "joints": [
            {
                "name": j.name,
                "level": j.level,
                "primitives": list(j.primitives),
                "coordinates": list(j.coordinates),
                "isb_axes": list(j.isb_axes),
                "isb_signs": list(j.isb_signs),
                "soft_limits_isb_deg": [list(p) for p in j.soft_limits_isb_deg or ()],
            }
            for j in V2_TRUNK.joints
        ],
        "lumbar_body": LUMBAR_BODY,
        "l5s1_in_pelvis_m": list(locations.l5s1_in_pelvis_m),
        "t12l1_in_pelvis_m": list(locations.t12l1_in_pelvis_m),
        "locations_provenance": locations.provenance,
        "locations_evidence": json.loads(json.dumps(locations.evidence or {})),
        "neutral_rotation": (
            "identity: primitives act about the model axes; the neutral "
            "lordosis/kyphosis is carried by the joint-centre chords (HRG-D2)"
        ),
        "hard_range_policy": HARD_RANGE_POLICY,
        **replaced["record"],
    }


def _move_hand_solids(
    club_body: dict[str, Any],
    target_body: dict[str, Any],
    suffixes: tuple[str, ...],
    club_to_target: Array,
) -> None:
    """Move solids with short names in ``suffixes`` from ``club_body`` to ``target_body``."""
    keep: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    for solid in club_body["solids"]:
        short = solid["name"].rsplit("/", 1)[-1]
        (moved if short in suffixes else keep).append(dict(solid))
    if {s["name"].rsplit("/", 1)[-1] for s in moved} != set(suffixes):
        raise ValueError(f"club body lacks hand solids {suffixes}")
    for solid in moved:
        solid["placement"] = (
            club_to_target @ np.asarray(solid["placement"], dtype=float)
        ).tolist()
    club_body["solids"] = keep
    target_body.setdefault("solids", []).extend(moved)


def _splice_v2_separated_elements(
    out: dict[str, Any], left_body: dict[str, Any], free_joint: dict[str, Any]
) -> None:
    """Insert left_body, free_joint, and free coordinates into upper-body slices."""
    counts = dict(out["upper_body_counts"])
    n_bodies = int(counts["bodies"])
    n_joints = int(counts["joints"])
    n_coords = int(counts["coordinates"])
    out["bodies"] = [
        *out["bodies"][:n_bodies],
        left_body,
        *out["bodies"][n_bodies:],
    ]
    out["joints"] = [
        *out["joints"][:n_joints],
        free_joint,
        *out["joints"][n_joints:],
    ]
    out["coordinate_order"] = [
        *out["coordinate_order"][:n_coords],
        *CLUB_FREE_COORDINATES,
        *out["coordinate_order"][n_coords:],
    ]
    counts["bodies"] = n_bodies + 1
    counts["joints"] = n_joints + 1
    counts["coordinates"] = n_coords + len(CLUB_FREE_COORDINATES)
    out["upper_body_counts"] = counts
    out["passive_coordinates"] = list(CLUB_FREE_COORDINATES)


def separate_v2_hands(v2_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Separate ``LHandBody`` and ``RHandStandoff`` from the bare club in a ``full-body-v2`` document.

    Opt-in topology transform for HRG-GS-G11 (#13309):
    - ``LHand`` and ``LHandStandoff`` solids move onto ``LEFT_HAND_BODY`` (child of
      the left wrist joint, keeping ``child_to_follower`` so the body frame coincides
      with the original club frame on the left-wrist chain).
    - ``RHand`` solid moves onto ``RHandStandoff`` (child of the right wrist joint)
      via ``placement_a @ inv(placement_b)``.
    - The bare club body (``closure["body_b"]``) is rooted at ``"world"`` through
      the 6-DOF ``CLUB_FREE_JOINT`` with neutral ``parent_to_base`` equal to its
      welded neutral world pose, so all body, solid, and frame neutral poses match.
    - Populates ``grip_bushing`` and ``hand_separation`` metadata (including the
      marker-unobservability identifiability note).
    """
    require_full_body_schema(
        v2_spec, consumer="separate_v2_hands", accepted=(FULL_BODY_V2_SCHEMA_VERSION,)
    )
    if v2_spec.get("hand_separation", {}).get("enabled"):
        raise ValueError("v2 document already has separated hands")
    out = json.loads(json.dumps(v2_spec))
    neutral_poses = _neutral_body_poses(out)
    interface = GripInterface.from_spec(out)
    closure = out["closure"]
    club_name, right_name = closure["body_b"], closure["body_a"]
    club_world = neutral_poses[club_name]

    joints_to_club = [j for j in out["joints"] if j["child"] == club_name]
    if len(joints_to_club) != 1:
        raise ValueError("expected exactly one joint parenting the club")
    left_joint = joints_to_club[0]
    pa = np.asarray(closure["placement_a"], dtype=float)
    pb = np.asarray(closure["placement_b"], dtype=float)
    club_to_right = pa @ np.linalg.inv(pb)

    bodies_by_name = {b["name"]: b for b in out["bodies"]}
    club_body = bodies_by_name[club_name]
    right_body = bodies_by_name[right_name]
    left_body: dict[str, Any] = {"name": LEFT_HAND_BODY, "solids": []}
    _move_hand_solids(club_body, left_body, _LEFT_HAND_SOLIDS, np.eye(4))
    _move_hand_solids(club_body, right_body, _RIGHT_HAND_SOLIDS, club_to_right)

    left_joint["child"] = LEFT_HAND_BODY
    free_joint = JointSpec(
        CLUB_FREE_JOINT,
        "world",
        club_name,
        club_world.tolist(),
        np.eye(4).tolist(),
        _FREE_PRIMITIVES,
        CLUB_FREE_COORDINATES,
    ).as_document()

    for frame in out["frames"]:
        if frame["body"] == club_name and frame["name"] == "Grip_LeftHand":
            frame["body"] = LEFT_HAND_BODY

    _splice_v2_separated_elements(out, left_body, free_joint)

    g_left, g_right = interface.left.matrix(), interface.right.matrix()
    out["grip_bushing"] = {
        "L": {
            "hand_body": LEFT_HAND_BODY,
            "hand_frame": g_left.tolist(),
            "club_body": club_name,
            "club_frame": g_left.tolist(),
        },
        "R": {
            "hand_body": right_name,
            "hand_frame": (club_to_right @ g_right).tolist(),
            "club_body": club_name,
            "club_frame": g_right.tolist(),
        },
    }
    out["hand_separation"] = {
        "enabled": True,
        "mode": "separated_v2",
        "left_hand_body": LEFT_HAND_BODY,
        "right_hand_body": right_name,
        "club_body": club_name,
        "free_joint": CLUB_FREE_JOINT,
        "free_coordinates": list(CLUB_FREE_COORDINATES),
        "identifiability_note": HAND_HAND_IDENTIFIABILITY_NOTE,
    }
    out["upper_body_sha256"] = canonical_sha256(upper_body_slice(out))
    validate_full_body_v2_spec(out, upper_body_slice(out))
    return out


def derive_full_body_v2_spec(
    v1: Mapping[str, Any],
    locations: TrunkJointLocations,
    *,
    provenance: str,
    separate_hands: bool = False,
) -> dict[str, Any]:
    """Derive a NEW ``full-body-v2`` document from a v1 anthropometric document.

    ``v1`` is validated against its own upper-body slice and never modified;
    its identity is recorded as ``source_spec_sha256``. Every retained body,
    solid and frame keeps its neutral (q = 0) world pose. When ``separate_hands``
    is ``True`` (explicit opt-in, default ``False``), :func:`separate_v2_hands`
    separates ``LHandBody`` and ``RHandStandoff`` from the bare club body.

    Raises ``ValueError`` for a non-v1 document, a native spec without the de
    Leva trunk solids, a referenced dummy body or empty provenance, and
    ``TypeError`` for ``locations`` of the wrong type. Postcondition: the
    result passes :func:`validate_full_body_v2_spec` against its own slice.
    """
    require_full_body_schema(
        v1, consumer="derive_full_body_v2_spec", accepted=("full-body-v1",)
    )
    if not isinstance(locations, TrunkJointLocations):
        raise TypeError("locations must be TrunkJointLocations")
    if not provenance.strip():
        raise ValueError("A v2 document needs a provenance")
    # Anthropometric documents change the upper body on purpose and so fail
    # the strict v1 identity check against any base; the structure must hold.
    validate_full_body_structure(v1)
    _check_dummy_is_free(v1, trunk_bodies(v1)["twist_base"])
    replaced = _replace_trunk(v1, locations)
    trunk_names = set(V1_TRUNK.coordinates)
    trunk_names |= set(V2_TRUNK.coordinates)
    ranges = {
        k: v
        for k, v in dict(v1.get("coordinate_ranges_deg", {})).items()
        if k not in trunk_names
    }
    counts = dict(v1["upper_body_counts"])
    counts["coordinates"] += len(V2_TRUNK.coordinates) - len(V1_TRUNK.coordinates)
    document = {
        **json.loads(json.dumps(v1)),
        "schema_version": FULL_BODY_V2_SCHEMA_VERSION,
        "upper_body_schema_version": V2_UPPER_BODY_SCHEMA_VERSION,
        "upper_body_qualification": (
            "full-body-v2 regional trunk (HRG-D1 schema); unqualified"
        ),
        "upper_body_counts": counts,
        "base_upper_body_sha256": v1["upper_body_sha256"],
        "source_spec_sha256": canonical_sha256(v1),
        "coordinate_order": v2_coordinate_order(v1["coordinate_order"]),
        "coordinate_ranges_deg": ranges,
        "soft_limits_deg": {k: list(v) for k, v in v2_trunk_ranges_deg().items()},
        "bodies": replaced["bodies"],
        "joints": replaced["joints"],
        "frames": replaced["frames"],
        "trunk_topology": _trunk_record(locations, replaced),
        "provenance": provenance,
        "qualification": (
            "full-body-v2 schema fixture (HRG-D1); geometry and dynamics "
            "unqualified until HRG-D2/Q1"
        ),
    }
    document["upper_body_sha256"] = canonical_sha256(upper_body_slice(document))
    validate_full_body_v2_spec(document, upper_body_slice(document))
    if separate_hands:
        return separate_v2_hands(document)
    return document


def _check_v2_trunk(document: Mapping[str, Any]) -> None:
    try:
        joints = V2_TRUNK.locate(document)
    except ValueError as exc:
        raise ValueError(f"Invalid full-body-v2 trunk: {exc}") from exc
    if joints[0]["parent"] != _pelvis(document) or joints[0]["child"] != LUMBAR_BODY:
        raise ValueError("full-body-v2 trunk must run pelvis -> trunk_lumbar")
    order = list(document["coordinate_order"])
    if set(V1_TRUNK.coordinates) & set(order):
        raise ValueError("full-body-v2 trunk must not carry v1 trunk coordinates")
    start = order.index(V2_TRUNK.coordinates[0])
    if tuple(order[start : start + 6]) != V2_TRUNK.coordinates:
        raise ValueError("full-body-v2 trunk coordinates must be contiguous, in order")
    record = document.get("trunk_topology")
    if not isinstance(record, Mapping) or [
        c for j in record.get("joints", ()) for c in j["coordinates"]
    ] != list(V2_TRUNK.coordinates):
        raise ValueError("full-body-v2 trunk_topology record disagrees with the schema")


def validate_full_body_v2_spec(
    document: Mapping[str, Any], upper: Mapping[str, Any]
) -> None:
    """Raise ValueError unless ``document`` is a valid ``full-body-v2`` document.

    ``upper`` must be the document's own upper-body slice (recorded as
    ``upper_body_sha256``). The structure rules are the v1 ones
    (:func:`validate_full_body_structure`); in addition the trunk must be
    exactly :data:`V2_TRUNK` and both parent identities must be recorded.
    """
    if document.get("schema_version") != FULL_BODY_V2_SCHEMA_VERSION:
        raise ValueError("Not a full-body-v2 document")
    if document.get("upper_body_schema_version") != V2_UPPER_BODY_SCHEMA_VERSION:
        raise ValueError("full-body-v2 upper body must carry the v2 upper label")
    for key in ("source_spec_sha256", "base_upper_body_sha256"):
        if not _SHA256.match(str(document.get(key, ""))):
            raise ValueError(f"full-body-v2 needs a 64-hex {key}")
    validate_upper_body_identity(document, upper)
    validate_full_body_structure(document)
    _check_v2_trunk(document)


FULL_BODY_SCHEMA = FullBodySchema(
    schema_id=FULL_BODY_V2_SCHEMA_VERSION,
    trunk=V2_TRUNK,
    validate=validate_full_body_v2_spec,
    description="regional lumbar (L5/S1) and thoracic (T12/L1) trunk joints",
)
