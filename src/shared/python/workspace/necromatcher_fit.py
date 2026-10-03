"""Validation of source-bound, unqualified native coordinate samples.

These records preserve research results, not calibrated motion or controls.
Coordinate units are declared by the producer; model compilation and scientific
acceptance are separate requirements. Source frame identities remain exact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary

FIT_SCHEMA = "necromatcher/kinematic-fit/1"
_FIELDS = frozenset(
    {
        "schema_version",
        "qualification",
        "physical_time_qualified",
        "dynamics_replayed",
        "model_id",
        "model_hash",
        "capture_id",
        "capture_hash",
        "coordinate_order",
        "coordinate_units",
        "frame_indices",
        "frames",
        "q",
        "provenance",
        "evidence",
    }
)


def read_kinematic_fit(
    source: Path, library: NecromatcherLibrary, swing_id: str
) -> dict[str, Any]:
    """Verify bytes, parent versions and exact source frames before publication."""
    from .necromatcher_review import CaptureReview

    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("Fit must contain exactly the kinematic-fit schema fields")
    if (
        payload["schema_version"] != FIT_SCHEMA
        or payload["qualification"] != "monocular_research_hypothesis"
        or payload["physical_time_qualified"] is not False
        or payload["dynamics_replayed"] is not False
    ):
        raise ValueError("Fit storage accepts only unqualified kinematic research")
    for name, kind in (("model", "native_model"), ("capture", "image_capture")):
        if not isinstance(payload[f"{name}_id"], str):
            raise ValueError("Fit parent identity must be a string")
        asset = library.load_asset(payload[f"{name}_id"])
        if asset.kind != kind or asset.session_id != swing_id:
            raise ValueError("Fit parents must belong to the same swing session")
        if payload[f"{name}_hash"] != asset.metadata["hash"]:
            raise ValueError("Fit parent hash mismatch")
        if name == "model" and payload["coordinate_order"] != asset.metadata["dofs"]:
            raise ValueError("Fit coordinate order must match its model version")
    order, units = payload["coordinate_order"], payload["coordinate_units"]
    if (
        not isinstance(units, list)
        or len(units) != len(order)
        or any(not isinstance(unit, str) or unit not in {"rad", "m"} for unit in units)
    ):
        raise ValueError("Fit coordinates require declared rad or m units")
    indices, frames = payload["frame_indices"], payload["frames"]
    if (
        not isinstance(indices, list)
        or len(indices) < 2
        or any(type(index) is not int or index < 0 for index in indices)
        or any(b <= a for a, b in zip(indices, indices[1:], strict=False))
        or not isinstance(frames, list)
        or len(frames) != len(indices)
    ):
        raise ValueError(
            "Fit frame indices must be increasing with matching identities"
        )
    try:
        coordinates = np.asarray(payload["q"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Fit coordinates must be a finite numeric matrix") from exc
    if (
        coordinates.dtype.kind not in "ifu"
        or coordinates.shape != (len(indices), len(order))
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("Fit coordinates must be finite in frame and model order")
    provenance = payload["provenance"]
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("description"), str)
        or not provenance["description"].strip()
        or not isinstance(payload["evidence"], dict)
    ):
        raise ValueError("Fit requires explicit provenance and research evidence")
    # Reject non-finite auxiliary evidence as well; retain the original JSON bytes.
    json.dumps(payload, allow_nan=False)
    with CaptureReview(library, payload["capture_id"]) as review:
        for index, identity in zip(indices, frames, strict=True):
            if index >= review.frame_count or identity != review.frame(index)["frame"]:
                raise ValueError("Fit source frame identity mismatch")
    from .necromatcher_placement import validate_placement_lineage

    validate_placement_lineage(payload, library, swing_id)
    from .necromatcher_hypothesis import validate_hypothesis_seed

    validate_hypothesis_seed(library, payload)
    return payload
