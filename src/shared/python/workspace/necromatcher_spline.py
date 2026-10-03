"""Read-only saved-spline identities shared by plans and native workers."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

import numpy as np
from src.shared.python.estimation import CubicHermiteSplineTrajectory
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


def verify_preserved_fit_samples(
    source: Mapping[str, Any], start: ImageSplineStart
) -> np.ndarray:
    """Validate stored dense poses through the canonical Hermite provider."""
    times = np.array(
        [
            frame["pts_ticks"]
            * frame["timebase_numerator"]
            / frame["timebase_denominator"]
            for frame in source["frames"]
        ]
    )
    if np.any(times < start.knot_times[0]) or np.any(times > start.knot_times[-1]):
        raise ValueError("Parent source clock exceeds preserved spline interval")
    trajectory = CubicHermiteSplineTrajectory(
        np.asarray(start.knot_times), len(start.free_coordinates)
    )
    free = trajectory.evaluate(np.asarray(start.spline_coefficients), times).q
    samples = np.asarray(source["q"], dtype=float)
    expected = np.tile(samples[0], (len(samples), 1))
    indices = [start.coordinate_order.index(name) for name in start.free_coordinates]
    expected[:, indices] = free
    if not np.allclose(samples, expected, rtol=1e-8, atol=1e-10):
        raise ValueError("Parent samples disagree with the preserved canonical spline")
    return expected
