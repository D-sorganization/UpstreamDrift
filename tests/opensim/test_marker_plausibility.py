"""Unit tests for marker and trunk plausibility gates (#13434).

No OpenSim imports allowed; all tests use synthetic priors and offsets.
"""

from __future__ import annotations

import math

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.anatomical_marker_priors import (
    AnatomicalPrior,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_plausibility import (
    ImplausibleInputError,
    MarkerCheck,
    PlausibilityReport,
    TrunkCheck,
    assess_marker_placements,
    assess_trunk_rotation,
    require_plausible,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_priors() -> dict[str, AnatomicalPrior]:
    return {
        "RKneeOut": AnatomicalPrior(
            body="femur_r",
            offset_m=(0.0, -0.40, 0.05),
            radius_m=0.04,
            landmark="lateral femoral epicondyle right",
        ),
        "LKneeOut": AnatomicalPrior(
            body="femur_l",
            offset_m=(0.0, -0.40, -0.05),
            radius_m=0.04,
            landmark="lateral femoral epicondyle left",
        ),
        "WaistRight": AnatomicalPrior(
            body="pelvis",
            offset_m=(-0.02, 0.04, 0.17),
            radius_m=0.06,
            landmark="waist belt right",
        ),
    }


@pytest.fixture
def passing_offsets() -> dict[str, tuple[str, tuple[float, float, float]]]:
    return {
        "RKneeOut": ("femur_r", (0.0, -0.40, 0.05)),
        "LKneeOut": ("femur_l", (0.0, -0.40, -0.05)),
        "WaistRight": ("pelvis", (-0.02, 0.04, 0.17)),
    }


def test_passing_set(
    synthetic_priors: dict[str, AnatomicalPrior],
    passing_offsets: dict[str, tuple[str, tuple[float, float, float]]],
) -> None:
    checks = assess_marker_placements(passing_offsets, synthetic_priors)
    report = PlausibilityReport(checks=checks)
    assert report.passed
    assert report.failures() == ()
    assert len(checks) == 3
    assert all(c.passed for c in checks)
    require_plausible(report)


def test_marker_moved_beyond_radius_fails(
    synthetic_priors: dict[str, AnatomicalPrior],
    passing_offsets: dict[str, tuple[str, tuple[float, float, float]]],
) -> None:
    offsets = dict(passing_offsets)
    offsets["RKneeOut"] = ("femur_r", (0.0, -0.40, 0.15))
    checks = assess_marker_placements(offsets, synthetic_priors)
    report = PlausibilityReport(checks=checks)

    assert not report.passed
    assert "RKneeOut" in report.failures()
    rknee = next(c for c in checks if c.name == "RKneeOut")
    assert not rknee.passed
    assert rknee.deviation_m > rknee.radius_m


def test_left_marker_with_positive_z_fails_side_ok(
    synthetic_priors: dict[str, AnatomicalPrior],
    passing_offsets: dict[str, tuple[str, tuple[float, float, float]]],
) -> None:
    offsets = dict(passing_offsets)
    offsets["LKneeOut"] = ("femur_l", (0.0, -0.40, 0.02))
    checks = assess_marker_placements(offsets, synthetic_priors)
    report = PlausibilityReport(checks=checks)

    assert not report.passed
    assert "LKneeOut" in report.failures()
    lknee = next(c for c in checks if c.name == "LKneeOut")
    assert not lknee.side_ok
    assert not lknee.passed


def test_wrong_body_fails(
    synthetic_priors: dict[str, AnatomicalPrior],
    passing_offsets: dict[str, tuple[str, tuple[float, float, float]]],
) -> None:
    offsets = dict(passing_offsets)
    offsets["LKneeOut"] = ("tibia_l", (0.0, -0.40, -0.05))
    checks = assess_marker_placements(offsets, synthetic_priors)
    report = PlausibilityReport(checks=checks)

    assert not report.passed
    assert "LKneeOut" in report.failures()
    lknee = next(c for c in checks if c.name == "LKneeOut")
    assert not lknee.passed


def test_missing_marker_fails(
    synthetic_priors: dict[str, AnatomicalPrior],
    passing_offsets: dict[str, tuple[str, tuple[float, float, float]]],
) -> None:
    offsets = {k: v for k, v in passing_offsets.items() if k != "WaistRight"}
    checks = assess_marker_placements(offsets, synthetic_priors)
    report = PlausibilityReport(checks=checks)

    assert not report.passed
    assert "WaistRight" in report.failures()
    waist = next(c for c in checks if c.name == "WaistRight")
    assert not waist.passed
    assert math.isinf(waist.deviation_m)


def test_trunk_20_50_40_passes() -> None:
    rotations = {"address": 20.0, "top": 50.0, "impact": 40.0}
    checks = assess_trunk_rotation(rotations)
    report = PlausibilityReport(trunk=checks)

    assert report.passed
    assert len(checks) == 3
    assert all(t.passed for t in checks)
    assert report.failures() == ()
    require_plausible(report)


def test_trunk_65_4_113_4_120_5_fails_all_three() -> None:
    rotations = {"address": 65.4, "top": 113.4, "impact": 120.5}
    checks = assess_trunk_rotation(rotations)
    report = PlausibilityReport(trunk=checks)

    assert not report.passed
    assert len(checks) == 3
    assert all(not t.passed for t in checks)
    assert set(report.failures()) == {"address", "top", "impact"}


def test_require_plausible_raises_with_every_failure_named() -> None:
    checks = (
        MarkerCheck("RKneeOut", "femur_r", 0.10, 0.04, True, False),
        MarkerCheck("LKneeOut", "femur_l", 0.02, 0.04, False, False),
        MarkerCheck("WaistRight", "pelvis", 0.01, 0.06, True, True),
    )
    trunk = (
        TrunkCheck("address", 65.4, 30.0, False),
        TrunkCheck("top", 113.4, 75.0, False),
        TrunkCheck("impact", 40.0, 60.0, True),
    )
    report = PlausibilityReport(checks=checks, trunk=trunk)

    with pytest.raises(ImplausibleInputError) as exc_info:
        require_plausible(report)

    err_msg = str(exc_info.value)
    for failure in ("RKneeOut", "LKneeOut", "address", "top"):
        assert failure in err_msg, f"Expected {failure} named in error: {err_msg}"
    assert "WaistRight" not in err_msg
    assert "impact" not in err_msg


def test_nan_rotation_raises_value_error() -> None:
    with pytest.raises(ValueError):
        assess_trunk_rotation({"address": float("nan"), "top": 50.0, "impact": 40.0})
    with pytest.raises(ValueError):
        assess_trunk_rotation({"address": 20.0, "top": float("nan"), "impact": 40.0})
    with pytest.raises(ValueError):
        assess_trunk_rotation({"address": 20.0, "top": 50.0, "impact": float("nan")})
