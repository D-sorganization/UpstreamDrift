"""Club sole contact at the fitted address pose (#12969)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.motion_matching import club_face_target as cft
from src.shared.python.motion_matching.pipeline.constants import (
    CLUB_SOLE_GROUND_TOLERANCE_M,
)
from src.shared.python.motion_matching.tour_capture_contract import MARKER_SEGMENTS


def lane_has_club_markers(labels: tuple[str, ...] | list[str]) -> bool:
    """True when the capture lane tracks at least one club marker."""
    club = set(MARKER_SEGMENTS["club"])
    return any(label in club for label in labels)


def club_sole_height_m(
    kin: Any, q: np.ndarray, spec: Mapping[str, Any], ground: Any
) -> float | None:
    """Rendered sole height above ``ground`` at ``q``, or ``None`` without a club."""
    try:
        cft.require_club_assembly(spec)
    except ValueError:
        return None
    sole = cft.model_club_soles(kin, np.asarray(q, dtype=float)[None, :], spec)[0]
    return cft.sole_height_above_ground_m(sole, ground)


def club_sole_report(
    kin: Any, q: np.ndarray, spec: Mapping[str, Any], ground: Any
) -> dict[str, Any]:
    """Receipt block for club sole height at address."""
    height_m = club_sole_height_m(kin, q, spec, ground)
    if height_m is None:
        return {"available": False, "reason": "spec has no club body"}
    return {
        "available": True,
        "measurement_point": "club_body_origin",
        "height_above_ground_m": height_m,
        "tolerance_m": CLUB_SOLE_GROUND_TOLERANCE_M,
        "within_tolerance": height_m <= CLUB_SOLE_GROUND_TOLERANCE_M,
    }


def assert_club_sole_at_ground(
    kin: Any, q: np.ndarray, spec: Mapping[str, Any], ground: Any
) -> None:
    """DbC: the address pose rests the club sole on the ground within tolerance."""
    report = club_sole_report(kin, q, spec, ground)
    if not report.get("available"):
        return
    height_m = float(report["height_above_ground_m"])
    if height_m > CLUB_SOLE_GROUND_TOLERANCE_M:
        raise ValueError(
            "club sole is "
            f"{height_m * 1e3:.1f} mm above ground; "
            f"limit is {CLUB_SOLE_GROUND_TOLERANCE_M * 1e3:.1f} mm"
        )
