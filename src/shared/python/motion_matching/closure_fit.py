"""Fit the two-hand closure weld from a fitted pose (MM-2, #10104).

The dual-grip closure welds a frame on the trail hand (``placement_a`` in
``body_a``) to a frame on the club (``placement_b`` in ``body_b``). The
native document fixes that relation from the Simscape model; for the
anthropometric documents the relation must instead follow the golfer's
address: with the wrists at anatomical values and the closure released,
the trail hand sits where the arm chain puts it, and the club where its
markers put it. ``fit_closure_placement`` rewrites ``placement_b`` so the
weld holds exactly at that pose. Engines keep the weld semantics; only the
numbers change, so cross-engine parity is untouched.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Pose = tuple[Array, Array]  # (rotation 3x3, translation 3) of a frame in the world


def _homogeneous(pose: Pose) -> Array:
    rotation, translation = pose
    r = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all():
        raise ValueError("A pose is a finite 3x3 rotation and a 3-vector")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-9) or np.linalg.det(r) < 0:
        raise ValueError("Pose rotation must be proper orthonormal")
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def fit_closure_placement(
    document: Mapping[str, Any], hand_frame_world: Pose, club_body_world: Pose
) -> dict[str, Any]:
    """Return a copy of ``document`` whose closure ``placement_b`` makes the
    weld hold at the given world poses of the hand-side closure frame
    (``body_a`` with ``placement_a`` applied) and of the club body frame.

    Postcondition: ``club_body_world @ placement_b_new`` equals
    ``hand_frame_world``; the closure keeps its bodies, name and
    ``placement_a``; the document records the fit under ``closure_fit``.
    """
    doc = json.loads(json.dumps(document))
    closure = doc.get("closure")
    if not closure or "placement_b" not in closure:
        raise ValueError("Document has no closure with placement_b")
    hand = _homogeneous(hand_frame_world)
    club = _homogeneous(club_body_world)
    placement_b = np.linalg.inv(club) @ hand
    old = np.asarray(closure["placement_b"], dtype=float)
    delta_rot = old[:3, :3].T @ placement_b[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(delta_rot) - 1) / 2, -1, 1))))
    shift = float(np.linalg.norm(placement_b[:3, 3] - old[:3, 3]))
    closure["placement_b"] = placement_b.tolist()
    doc["closure_fit"] = {
        "method": "trail hand frame and club body poses at the fitted address, wrists at anatomical values, closure released",
        "rotation_change_deg": angle,
        "translation_change_m": shift,
    }
    return doc


def closure_residual(
    hand_frame_world: Pose, club_body_world: Pose, placement_b: Any
) -> tuple[float, float]:
    """Position (m) and rotation (rad) mismatch of a weld at world poses."""
    hand = _homogeneous(hand_frame_world)
    predicted = _homogeneous(club_body_world) @ np.asarray(placement_b, dtype=float)
    pos = float(np.linalg.norm(predicted[:3, 3] - hand[:3, 3]))
    rel = hand[:3, :3].T @ predicted[:3, :3]
    rot = float(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0)))
    return pos, rot
