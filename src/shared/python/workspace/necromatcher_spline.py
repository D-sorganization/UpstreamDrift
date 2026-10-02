"""Read-only saved-spline identities shared by plans and native workers."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

from src.shared.python.motion_matching.historical_fit import ImageSplineStart


def preserved_fit_spline(fit: Mapping[str, Any]) -> ImageSplineStart | None:
    """Recall exact coefficients; absent legacy splines are unavailable, corrupt ones reject.

    This validates serialized identity without compiling an engine. Native execution
    independently binds the result to the compiled plant and all parent poses.
    """
    original = fit.get("evidence", {}).get("original_fit", {})
    if not isinstance(original, Mapping):
        raise ValueError("Saved fit evidence must contain an original-fit object")
    fields = {
        "knot_times",
        "spline_coefficients",
        "coordinate_order",
        "free_coordinates",
    }
    declared = "spline_start" in original
    spline_fields = fields & set(original)
    # Legacy camera/attachment records may contain coordinate identities alone.
    if not declared and not {"knot_times", "spline_coefficients"} & spline_fields:
        return None
    if not fields.issubset(original):
        raise ValueError("Incomplete preserved spline record")
    try:
        definition = fit["provenance"]["native_definition"]
        model_sha = hashlib.sha256(
            json.dumps(definition, allow_nan=False).encode("utf-8")
        ).hexdigest()
        order = tuple(fit["coordinate_order"])
        start = (
            ImageSplineStart.from_record(original["spline_start"])
            if declared
            else ImageSplineStart.from_coefficients(
                original["knot_times"],
                original["spline_coefficients"],
                tuple(original["coordinate_order"]),
                tuple(original["free_coordinates"]),
                model_sha,
            )
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Preserved spline lacks bound native model identity") from exc
    if (
        start.model_sha != model_sha
        or start.coordinate_order != order
        or start.coordinate_order != tuple(original["coordinate_order"])
        or start.free_coordinates != tuple(original["free_coordinates"])
        or start.knot_times != tuple(original["knot_times"])
        or start.spline_coefficients != tuple(original["spline_coefficients"])
    ):
        raise ValueError("Preserved spline identity differs from bound parent record")
    return start
