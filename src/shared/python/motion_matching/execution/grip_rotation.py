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
