"""Set a spec document's fitted hand rotation per capture (#12221).

The constant hand rotation ``G(R)`` (``subject.grip_rotation_deg``) is composed
into each wrist joint's ``child_to_follower`` by the builder. A per-capture
refit (``fit_grip_rotation.py``) yields a total ``T``; this module moves a
committed document from ``R`` to ``T`` in place of a rebuild, which would
discard its hand-tuned contact parameters:
``c2f <- c2f @ G(R).T @ G(T)`` and ``grip_rotation_deg[side] = T``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.model_appearance.grip_pose import GripStyle
from src.shared.python.model_appearance.schema import physics_spec_sha256
from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    HandAngles,
    grip_spec_to_physical,
    to_grip_rotation,
)
from src.shared.python.motion_matching.grip_fit import grip_rotation

#: Hand body of each side (see ``anthropometric_geometry``): the lead (left)
#: hand is the club, the trail (right) hand the standoff body.
HAND_BODY_SUFFIX = {"L": "Clubface Vector", "R": "RHandStandoff"}


def _target(side: str, value: Sequence[float]) -> list[float]:
    if side not in HAND_BODY_SUFFIX:
        raise ValueError(
            f"side must be one of {sorted(HAND_BODY_SUFFIX)}, got {side!r}"
        )
    angles = np.asarray(value, dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError(f"grip rotation of {side} must be three finite angles")
    return [float(v) for v in angles]


def _wrist_joint(document: Mapping[str, Any], side: str) -> dict[str, Any]:
    suffix = HAND_BODY_SUFFIX[side]
    for joint in document["joints"]:
        if joint["child"].endswith(suffix):
            return joint  # type: ignore[no-any-return]
    raise ValueError(f"document has no wrist joint whose child ends with {suffix}")


def apply_grip_rotation(
    document: Mapping[str, Any],
    totals_deg: Mapping[str, Sequence[float]],
    *,
    note: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return a copy of ``document`` whose hand rotation is ``totals_deg``.

    ``totals_deg`` maps ``"L"``/``"R"`` to extrinsic x-y-z degrees. Each wrist
    joint's rotation block becomes ``c2f @ G(R).T @ G(T)`` and
    ``subject.grip_rotation_deg`` records ``T``; the other side and the
    translation of ``child_to_follower`` are untouched. Raises ``ValueError``
    for an unknown side, a non-finite or mis-sized target, or a document
    without ``subject.grip_rotation_deg`` for the side. Postcondition: the
    result is unchanged, with ``changed`` False, when every target already
    equals the document's value, so applying twice equals applying once.
    ``note`` is appended to ``provenance`` when something changed.
    """
    targets = {side: _target(side, value) for side, value in totals_deg.items()}
    out = copy.deepcopy(dict(document))
    recorded = out.get("subject", {}).get("grip_rotation_deg", {})
    changed = False
    for side, total in targets.items():
        if side not in recorded:
            raise ValueError(f"document lacks subject.grip_rotation_deg for {side}")
        if np.allclose(recorded[side], total, rtol=0.0, atol=1e-9):
            continue
        joint = _wrist_joint(out, side)
        c2f = np.asarray(joint["child_to_follower"], dtype=float)
        current = grip_rotation(recorded[side])
        c2f[:3, :3] = c2f[:3, :3] @ current.T @ grip_rotation(total)
        joint["child_to_follower"] = c2f.tolist()
        recorded[side] = total
        changed = True
    if changed and note and note not in out.get("provenance", ""):
        out["provenance"] = out.get("provenance", "") + note
    return out, changed


#: Where each physical grip parameter came from (``subject.grip_provenance``).
GRIP_PROVENANCE_SOURCES = (
    "user_declared",
    "rom_set_identified",
    "default",
    "photo_estimate",
)
#: The physical grip parameters, in the key form used by the provenance map.
GRIP_PARAMETERS = tuple(
    [f"{hand}.{angle}" for hand in ("lead", "trail") for angle in HandAngles.FIELDS]
    + ["lead_below_butt_m", "hand_spacing_m"]
)
_OPTIONAL_GRIP_PARAMETERS = ("style", "trigger_finger")
_CLUB_SUFFIX = "Clubface Vector"
#: Solids of the club body that sit on a hand: (name suffix, hand).
_LEAD_SOLIDS = ("/LHand", "/LHandStandoff")
_TRAIL_SOLIDS = ("/RHand",)


def _provenance(value: str | Mapping[str, str]) -> dict[str, str]:
    """Per-parameter provenance from one source or a partial mapping."""
    if isinstance(value, str):
        value = dict.fromkeys(GRIP_PARAMETERS, value)
    else:
        allowed = set(GRIP_PARAMETERS) | set(_OPTIONAL_GRIP_PARAMETERS)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown grip provenance parameters: {unknown}")
        value = {**dict.fromkeys(GRIP_PARAMETERS, "user_declared"), **value}
    bad = sorted({v for v in value.values() if v not in GRIP_PROVENANCE_SOURCES})
    if bad:
        raise ValueError(
            f"grip provenance must be one of {GRIP_PROVENANCE_SOURCES}, got {bad}"
        )
    return dict(value)


def _shift_y(matrix: Any, delta: float) -> list[list[float]]:
    m = np.asarray(matrix, dtype=float).copy()
    m[1, 3] += delta
    return m.tolist()


def _shift_hands(out: dict[str, Any], lead: float, trail: float) -> None:
    """Move the hand solids and the ``LW`` frame on the club body by the shifts."""
    club = next(b for b in out["bodies"] if b["name"].endswith(_CLUB_SUFFIX))
    for solid in club["solids"]:
        for suffixes, delta in ((_LEAD_SOLIDS, lead), (_TRAIL_SOLIDS, trail)):
            if solid["name"].endswith(suffixes):
                solid["placement"] = _shift_y(solid["placement"], delta)
    for frame in out["frames"]:
        if frame["body"] == club["name"] and frame["name"] == "LW":
            frame["placement"] = _shift_y(frame["placement"], lead)


def _shift_closure(out: dict[str, Any], grip: GripSpec) -> None:
    """Place the lead wrist and the trail closure along the shaft.

    The lead wrist sits ``lead_below_butt_m`` below the butt, ``length_m`` from
    the head; the trail weld sits ``hand_spacing_m`` further toward the head, that is
    ``+y`` of the lead wrist in the club frame.  Absolute targets,
    so applying a configuration twice moves nothing.  Mass and inertia are
    untouched: only placements move, so nothing is counted twice.
    """
    length = out.get("club", {}).get("length_m")
    if length is None:
        raise ValueError("document lacks club.length_m to place the hands on")
    joint = _wrist_joint(out, "L")
    lead_y = float(joint["child_to_follower"][1][3])
    trail_y = float(out["closure"]["placement_b"][1][3])
    new_lead = -(float(length) - grip.lead_below_butt_m)
    new_trail = new_lead + grip.hand_spacing_m
    joint["child_to_follower"] = _shift_y(joint["child_to_follower"], new_lead - lead_y)
    out["closure"]["placement_b"] = _shift_y(
        out["closure"]["placement_b"], new_trail - trail_y
    )
    _shift_hands(out, new_lead - lead_y, new_trail - trail_y)
    if "wrist_to_head_m" in out["club"]:
        out["club"]["wrist_to_head_m"] = -new_lead


def grip_spec_of(document: Mapping[str, Any]) -> GripSpec:
    """The :class:`GripSpec` a configured document carries.

    ``ValueError`` for a document without ``subject.grip_config``.
    """
    subject = document.get("subject", {})
    cfg = subject.get("grip_config")
    if cfg is None:
        raise ValueError("document carries no subject.grip_config")
    visual = document.get("visual_hints", {}).get("grip", {})
    raw_style = cfg.get("style", visual.get("style", GripStyle.OVERLAP.value))
    style = GripStyle(raw_style)
    trigger_finger = bool(
        cfg.get("trigger_finger", visual.get("trigger_finger", False))
    )
    thumb = visual.get("lead_thumb_offset_deg")
    extra: dict[str, float] = {}
    if thumb is not None:
        extra["lead_thumb_offset_deg"] = float(thumb)
    return GripSpec(
        lead=HandAngles(**cfg["lead"]),
        trail=HandAngles(**cfg["trail"]),
        lead_below_butt_m=cfg["lead_below_butt_m"],
        hand_spacing_m=cfg["hand_spacing_m"],
        handedness=cfg["handedness"],
        style=style,
        trigger_finger=trigger_finger,
        **extra,
    )


def _set_visual(out: dict[str, Any], grip: GripSpec) -> None:
    hints = out.setdefault("visual_hints", {})
    visual_grip: dict[str, Any] = {
        "style": grip.style.value,
        "lead_thumb_offset_deg": grip.lead_thumb_offset_deg,
    }
    if grip.trigger_finger:
        visual_grip["trigger_finger"] = True
    hints["grip"] = visual_grip


def apply_grip_config(
    document: Mapping[str, Any],
    grip: GripSpec,
    *,
    provenance: str | Mapping[str, str] = "user_declared",
) -> dict[str, Any]:
    """New spec revision whose hands follow ``grip`` (HRG-GR2, #13203).

    The wrist ``child_to_follower`` rotation takes the exact golf-angle
    rotation (double precision, via :func:`apply_grip_rotation`), the lead
    wrist, the trail ``closure.placement_b`` and the hand solids move along the
    shaft to ``lead_below_butt_m`` / ``hand_spacing_m``, and ``spec_revision``
    records the parent canonical hash, version and configuration.
    ``subject.grip_config`` holds the physical configuration and
    ``subject.grip_provenance`` the source of each parameter.  Style and thumb
    are visual only (``visual_hints.grip``): changing only them mints no
    physics revision.  The input is never mutated, so v1 bytes and receipts
    stay as they were.

    ``provenance`` is one of :data:`GRIP_PROVENANCE_SOURCES` for every
    parameter, or a partial mapping (the rest ``user_declared``).
    Preconditions: a :class:`GripSpec`; a right-handed document (a mirrored
    ``child_to_follower`` for a left-handed physical model is not implemented,
    so ``ValueError`` rather than a silently right-handed model).
    Postcondition: applying the same configuration to the result returns an
    equal document.
    """
    if not isinstance(grip, GripSpec):
        raise TypeError("grip must be a GripSpec")
    if grip.handedness == "left" or subject_handedness(document) == "left":
        raise ValueError(
            "left-handed physical grip configuration is not implemented: the "
            "wrist child_to_follower would have to be mirrored"
        )
    prov = _provenance(provenance)
    physical = grip_spec_to_physical(grip)
    subject = document.get("subject", {})
    if (
        subject.get("grip_config") == physical
        and subject.get("grip_provenance") == prov
    ):
        out = copy.deepcopy(dict(document))  # visual-only restyle, same physics
        _set_visual(out, grip)
        return out
    totals: dict[str, Sequence[float]] = {}
    for side, euler in to_grip_rotation(grip).items():
        totals[side] = euler
    out, _ = apply_grip_rotation(document, totals)
    _shift_closure(out, grip)
    version = int(document.get("spec_revision", {}).get("version", 0)) + 1
    out["subject"]["grip_config"] = physical
    out["subject"]["grip_provenance"] = prov
    out["spec_revision"] = {
        "version": version,
        "configuration": "grip",
        "parent_spec_canonical_sha256": canonical_sha256(document),
        "parent_physics_spec_sha256": physics_spec_sha256(document),
    }
    out["provenance"] = (
        out.get("provenance", "")
        + f"; grip configuration revision {version} (HRG-GR2 #13203)"
    )
    _set_visual(out, grip)
    return out
