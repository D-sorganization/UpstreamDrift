"""Unit-labelled joint delivery/outcome covariance on supported populations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import numpy as np

FIELD_UNITS = {
    "shaft_rotation_deg": "degree",
    "path_deg": "degree; positive right",
    "speed_mps": "m/s",
    "contact_horizontal_m": "m; positive toe",
    "contact_vertical_m": "m; positive high",
    "shaft_lean_deg": "degree",
    "attack_angle_deg": "degree",
    "face_deg": "degree; positive right",
    "loft_deg": "degree",
    "contact_face_deg": "degree; positive right",
    "contact_loft_deg": "degree",
    "carry_x_m": "m; downrange",
    "carry_y_m": "m; positive right",
    "spin_rpm": "rpm",
    "launch_azimuth_deg": "degree; positive right",
    "spin_axis_tilt_deg": "degree; positive draw",
    "launch_elevation_deg": "degree",
    "ball_speed_mps": "m/s",
}


def joint_diagnostics(rows: Sequence[Any]) -> dict:
    """Covariance units are row-unit × column-unit; constant correlations null.

    Each pair uses supported rows with both observables defined. Pair counts
    are published; this matters for undefined zero-spin axis orientation.
    """
    fields = list(FIELD_UNITS)
    values = np.asarray(
        [[getattr(row, name) for name in fields] for row in rows], dtype=float
    )
    if values.shape != (len(rows), len(fields)) or len(rows) < 2:
        raise ValueError("joint diagnostics require at least two complete records")
    covariance = []
    correlation = []
    counts = []
    for i in range(len(fields)):
        covrow: list[float | None] = []
        corrrow: list[float | None] = []
        countrow: list[int] = []
        for j in range(len(fields)):
            mask = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            a, b = values[mask, i], values[mask, j]
            countrow.append(int(mask.sum()))
            if len(a) < 2:
                covrow.append(None)
                corrrow.append(None)
                continue
            covrow.append(float(np.cov(a, b, ddof=1)[0, 1]))
            corrrow.append(
                float(np.corrcoef(a, b)[0, 1])
                if np.std(a) > 1e-14 and np.std(b) > 1e-14
                else None
            )
        covariance.append(covrow)
        correlation.append(corrrow)
        counts.append(countrow)
    return {
        "n": len(rows),
        "fields": fields,
        "units": FIELD_UNITS,
        "covariance": covariance,
        "correlation": correlation,
        "pair_counts": counts,
        "conditioning": "supported_airborne_rows; pairwise defined observables",
        "covariance_units": "row unit multiplied by column unit",
        "undefined_correlation": None,
    }
