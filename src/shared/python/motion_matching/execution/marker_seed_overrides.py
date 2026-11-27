"""Marker attachment seed overrides for the anthropometric spec builder (#15296).

A seed override states where a capture marker sits on the model: a body or
frame and an offset in metres in that frame, measured from the frame origin or,
with ``anchor``, from the origin of the joint between the body and the anchor
(for example the neck joint for the upper back markers). Dropped markers get no
attachment, which is how every fit that reads ``marker_attachments`` leaves
them out. Seeds are starting values for a calibration, not measurements.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from src.shared.python.motion_matching.full_body_spec import MarkerAttachment
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    TOUR_CAPTURE,
    tracked_labels,
)

# Body markers; the club cluster comes from the club marker file instead.
SEEDABLE_LABELS = tuple(
    label for label in tracked_labels() if label not in MARKER_SEGMENTS["club"]
)
_KEYS = {"body", "offset_m", "anchor", "provenance"}


def validate_marker_seed_overrides(
    overrides: Mapping[str, Mapping[str, Any]], dropped: Iterable[str] = ()
) -> None:
    """Refuse an override or a dropped label a build cannot honour.

    Raises ``ValueError`` for a label that is not a body marker of the capture,
    an unknown key, a blank body or anchor, an offset that is not three finite
    numbers, a dropped label the capture does not have and a label that is both
    seeded and dropped.
    """
    dropped = tuple(dropped)
    if unknown := sorted(set(overrides) - set(SEEDABLE_LABELS)):
        raise ValueError(f"unknown marker seed labels: {unknown}")
    if unknown := sorted(set(dropped) - set(TOUR_CAPTURE.labels)):
        raise ValueError(f"dropped markers are not capture labels: {unknown}")
    if both := sorted(set(overrides) & set(dropped)):
        raise ValueError(f"markers both seeded and dropped: {both}")
    for label, seed in overrides.items():
        if extra := sorted(set(seed) - _KEYS):
            raise ValueError(f"marker seed {label} has unknown keys {extra}")
        if "anchor" in seed and not str(seed["anchor"] or "").strip():
            raise ValueError(f"anchor of marker seed {label} must name a body")
        try:
            MarkerAttachment(str(seed.get("body", "")), tuple(seed.get("offset_m", ())))
        except (TypeError, ValueError) as error:
            raise ValueError(f"marker seed {label}: {error}") from error


def _joint_origin(
    body: str,
    anchor: str,
    joints: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Origin, in ``body``'s frame, of the joint from ``body`` to ``anchor``."""
    placed = {frame["name"]: frame for frame in frames}
    parent = placed.get(body, {"body": body, "placement": np.eye(4)})
    child = placed.get(anchor, {"body": anchor})["body"]
    for joint in joints:
        if joint["parent"] == parent["body"] and joint["child"] == child:
            origin = np.asarray(joint["parent_to_base"], dtype=float)[:, 3]
            placement = np.asarray(parent["placement"], dtype=float)
            return np.asarray(np.linalg.solve(placement, origin)[:3])
    raise ValueError(f"no joint from {body!r} to anchor {anchor!r}")


def apply_marker_seed_overrides(
    seeds: Mapping[str, MarkerAttachment],
    overrides: Mapping[str, Mapping[str, Any]] | None,
    dropped: Iterable[str] | None,
    joints: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
) -> Mapping[str, MarkerAttachment]:
    """``seeds`` with ``overrides`` applied and ``dropped`` labels removed.

    ``joints`` and ``frames`` are the document's, for the anchors. With neither
    setting the result is ``seeds`` itself; ``seeds`` is never modified.
    Raises ``ValueError`` as :func:`validate_marker_seed_overrides` does, and
    for an anchor with no joint from the seed's body.
    """
    if not overrides and not dropped:
        return seeds
    overrides, dropped = overrides or {}, tuple(dropped or ())
    validate_marker_seed_overrides(overrides, dropped)
    out = {label: seed for label, seed in seeds.items() if label not in dropped}
    for label, seed in overrides.items():
        offset = np.asarray(seed["offset_m"], dtype=float)
        if "anchor" in seed:
            offset = offset + _joint_origin(
                seed["body"], seed["anchor"], joints, frames
            )
        x, y, z = (float(value) for value in offset)
        out[label] = MarkerAttachment(seed["body"], (x, y, z))
    return out
