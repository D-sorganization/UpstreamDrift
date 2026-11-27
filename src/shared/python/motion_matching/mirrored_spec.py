"""Left-handed full-body spec as the sagittal mirror of a right-handed one (#15682).

World and pelvis frames are x forward, y left, z up, so the reflection is
``M = diag(1, -1, 1)``. A body pose ``X_b`` maps to ``M X_b N_b``; the local
reflection ``N_b`` (``body_reflections``) only picks the mirrored frame
convention, and attached data follow (``Y -> N Y N``, ``p -> N p``,
``I -> N I N``). A joint ``X_c = X_p P J(q) C^-1`` gets ``P' = N_p P K`` and
``C' = N_c C K``, so ``J'(q') = K J(q) K``:

* rotation-only joints use ``K = -I``: every coordinate keeps its value under
  the swapped name, so limits, seeds and lead/trail angles carry over;
* the floating root needs ``K = M`` (the world frame is shared), negating
  ``TranslationInputY``, ``HipInputX`` and ``HipInputZ``.

Names swap sides, so the right wrist carries the club body (name unchanged)
and the closure welds ``LHandStandoff`` to it.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.model_appearance.club_assembly import CLUB_BODY_SUFFIX
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

Array = NDArray[np.float64]

#: Sagittal reflection of the world/pelvis frame (y is the golfer's left).
MIRROR: Array = np.diag([1.0, -1.0, 1.0])
_INVERSION: Array = -np.eye(3)
#: A body's lateral axis must be within 45 deg of world y.
_DOMINANCE = 1.0 / np.sqrt(2.0) + 1e-6

_TOKENS = re.compile(
    r"(?P<elbow>(?:Left|Right) Elbow Joint/Spherical Solid1?)"
    r"|Left|Right"
    r"|_[lr](?=$|[^a-z0-9])"
    r"|(?<![A-Z])[LR](?=[A-Z]|$)"
)
_SWAP = {"Left": "Right", "Right": "Left", "_l": "_r", "_r": "_l", "L": "R", "R": "L"}


def _swap_token(match: re.Match[str]) -> str:
    if match["elbow"]:
        # The native export names the right elbow solid "Spherical Solid1".
        side = "Right" if match["elbow"].startswith("Left") else "Left"
        return f"{side} Elbow Joint/Spherical Solid{'1' if side == 'Right' else ''}"
    return _SWAP[match.group(0)]


def mirror_name(name: str) -> str:
    """``name`` with its body side swapped; an involution.

    ``RHandStandoff -> LHandStandoff``, ``Left Wrist -> Right Wrist``,
    ``femur_r -> femur_l``, ``LWInputX -> RWInputX``, ``HubtoLS -> HubtoRS``;
    ``Clubface Vector`` and ``COMRod`` are unchanged.
    """
    return _TOKENS.sub(_swap_token, name)


def body_reflections(document: Mapping[str, Any]) -> dict[str, Array]:
    """Local reflection ``N_b`` of every body (and ``"world"``).

    ``N_b`` negates the body axis most aligned with world y at ``q = 0``;
    ``ValueError`` when that axis is ambiguous. The club negates its x: its
    frame contract (shaft on ``-y``, axis offset on ``+z``, ``club_assembly``)
    then holds for the mirror. The choice is the same for the mirrored
    document, so mirroring twice is the identity.
    """
    reflections = {}
    for name, pose in spec_body_poses(document).items():
        lateral = np.abs(pose[:3, :3].T @ np.array([0.0, 1.0, 0.0]))
        club = name.endswith(CLUB_BODY_SUFFIX)
        axis = 0 if club else int(np.argmax(lateral))
        if not club and lateral[axis] < _DOMINANCE:
            raise ValueError(f"body {name!r} has no dominant lateral axis at q = 0")
        reflection = np.eye(3)
        reflection[axis, axis] = -1.0
        reflections[name] = reflection
    return reflections


def _joint_reflection(joint: Mapping[str, Any]) -> Array:
    sliding = any(p["primitive"].startswith("P") for p in joint["primitives"])
    return MIRROR if sliding else _INVERSION


def _primitive_sign(kind: str, reflection: Array) -> float:
    """Sign of ``q'`` in ``K J(q) K = J(q')`` (``det K = -1``)."""
    k = float(reflection["xyz".index(kind[1]), "xyz".index(kind[1])])
    return k if kind[0] == "P" else -k


def negated_coordinates(document: Mapping[str, Any]) -> frozenset[str]:
    """Coordinates of ``document`` whose sign flips under the mirror."""
    return frozenset(
        p["coordinate"]
        for joint in document["joints"]
        for p in joint["primitives"]
        if _primitive_sign(p["primitive"], _joint_reflection(joint)) < 0
    )


def mirror_coordinates(
    document: Mapping[str, Any], q: Mapping[str, float]
) -> dict[str, float]:
    """Coordinates of the mirrored model that pose it as the mirror of ``q``.

    Keys are swapped with :func:`mirror_name`; :func:`negated_coordinates`
    change sign. ``ValueError`` for a name ``document`` does not declare.
    """
    unknown = set(q) - set(document["coordinate_order"])
    if unknown:
        raise ValueError(f"unknown coordinate(s): {sorted(unknown)}")
    negated = negated_coordinates(document)
    return {mirror_name(k): -v if k in negated else v for k, v in q.items()}


def _rigid(matrix: Any, left: Array, right: Array) -> list[list[float]]:
    """``diag(left, 1) @ T @ diag(right, 1)`` for a 4x4 rigid transform."""
    t = np.asarray(matrix, dtype=float)
    out = np.eye(4)
    out[:3, :3] = left @ t[:3, :3] @ right
    out[:3, 3] = left @ t[:3, 3]
    return out.tolist()


def _point(value: Any, reflection: Array) -> Any:
    return None if value is None else (reflection @ np.asarray(value)).tolist()


_NAME_KEYS = ("name", "body", "parent", "child", "body_a", "body_b")


def _renamed(item: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    """``item`` with its name fields side-swapped and ``fields`` replaced."""
    names = {key: mirror_name(item[key]) for key in _NAME_KEYS if key in item}
    return {**item, **names, **fields}


def _mirror_body(body: Mapping[str, Any], n: Array) -> dict[str, Any]:
    solids = [
        _renamed(
            solid,
            placement=_rigid(solid["placement"], n, n),
            com_m=_point(solid["com_m"], n),
            inertia_com_kg_m2=(n @ np.asarray(solid["inertia_com_kg_m2"]) @ n).tolist(),
        )
        for solid in body["solids"]
    ]
    return _renamed(body, solids=solids)


def _mirror_joint(joint: Mapping[str, Any], n: Mapping[str, Array]) -> dict[str, Any]:
    k = _joint_reflection(joint)
    return _renamed(
        joint,
        parent_to_base=_rigid(joint["parent_to_base"], n[joint["parent"]], k),
        child_to_follower=_rigid(joint["child_to_follower"], n[joint["child"]], k),
        primitives=[
            {**p, "coordinate": mirror_name(p["coordinate"])}
            for p in joint["primitives"]
        ],
    )


def _mirror_closure(closure: Mapping[str, Any], n: Mapping[str, Array]) -> Any:
    """Both weld frames take the closing body's reflection on the right."""
    n_a, n_b = n[closure["body_a"]], n[closure["body_b"]]
    return _renamed(
        closure,
        placement_a=_rigid(closure["placement_a"], n_a, n_a),
        placement_b=_rigid(closure["placement_b"], n_b, n_a),
    )


def _mirror_limits(
    table: Mapping[str, Any], negated: frozenset[str], *, is_range: bool
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in table.items():
        if name in negated:
            value = [-value[1], -value[0]] if is_range else -value
        out[mirror_name(name)] = value
    return out


def _mirror_attachments(
    document: Mapping[str, Any], out: dict[str, Any], n: Mapping[str, Array]
) -> None:
    """Markers, contact spheres and visual hints of ``document`` into ``out``."""
    frame_body = {f["name"]: f["body"] for f in document["frames"]}
    markers = {}
    for label, marker in document.get("marker_attachments", {}).items():
        body = frame_body.get(marker["body"], marker["body"])
        offset = _point(marker["offset_m"], n[body])
        markers[mirror_name(label)] = _renamed(marker, offset_m=offset)
    out["marker_attachments"] = markers
    out["contact"]["spheres"] = [
        _renamed(s, position_m=_point(s["position_m"], n[s["body"]]))
        for s in document["contact"]["spheres"]
    ]
    solid_body = {s["name"]: b["name"] for b in document["bodies"] for s in b["solids"]}
    hints = document.get("visual_hints", {})
    out["visual_hints"] = {
        **copy.deepcopy(hints),
        "capsule_radius_m": {
            mirror_name(k): v for k, v in hints.get("capsule_radius_m", {}).items()
        },
        "shapes": {
            mirror_name(k): {**v, "center_m": _point(v["center_m"], n[solid_body[k]])}
            for k, v in hints.get("shapes", {}).items()
        },
    }


def _mirror_subject(document: Mapping[str, Any], out: dict[str, Any]) -> None:
    subject = out.setdefault("subject", {})
    subject["handedness"] = "left"
    grip = document.get("subject", {}).get("grip_rotation_deg")
    if grip is not None:
        subject["grip_rotation_deg"] = {mirror_name(k): v for k, v in grip.items()}
    torsion = document.get("tibial_torsion_deg")
    if torsion is not None:
        out["tibial_torsion_deg"] = {
            {"l": "r", "r": "l"}.get(k, k): v for k, v in torsion.items()
        }


def mirror_spec_to_left_handed(document: Mapping[str, Any]) -> dict[str, Any]:
    """The left-handed golfer's spec: ``document`` mirrored through x-z.

    Precondition: ``document`` is right-handed (``ValueError`` otherwise).
    Postconditions: ``document`` is not mutated; ``subject.handedness`` is
    ``"left"``; the right wrist carries the club; posing the result at
    ``mirror_coordinates(document, q)`` gives ``M X_b(q) N_b`` for every body.
    Unlisted fields (club, anthropometry, upper-body hashes) are copied as is.
    """
    if subject_handedness(document) != "right":
        raise ValueError("mirror_spec_to_left_handed needs a right-handed spec")
    n = body_reflections(document)
    negated = negated_coordinates(document)
    out = copy.deepcopy(dict(document))
    out["bodies"] = [_mirror_body(b, n[b["name"]]) for b in document["bodies"]]
    out["joints"] = [_mirror_joint(j, n) for j in document["joints"]]
    out["frames"] = [
        _renamed(f, placement=_rigid(f["placement"], n[f["body"]], n[f["body"]]))
        for f in document["frames"]
    ]
    if "closure" in document:
        out["closure"] = _mirror_closure(document["closure"], n)
    out["coordinate_order"] = [mirror_name(c) for c in document["coordinate_order"]]
    for key, is_range in (("coordinate_ranges_deg", True), ("address_seed_deg", False)):
        if key in document:
            out[key] = _mirror_limits(document[key], negated, is_range=is_range)
    out["gravity_m_s2"] = _point(document["gravity_m_s2"], MIRROR)
    _mirror_attachments(document, out, n)
    _mirror_subject(document, out)
    out["handedness_mirror"] = {
        "plane": "x-z (y lateral)",
        "source_sha256": canonical_sha256(document),
        "negated_coordinates": sorted(mirror_name(c) for c in negated),
    }
    out["provenance"] = (
        f"{document.get('provenance', '')}; mirrored to left-handed through the "
        "sagittal x-z plane (#15682)"
    )
    if len({b["name"] for b in out["bodies"]}) != len(out["bodies"]):
        raise ValueError("mirrored body names collide")
    return out
