"""1-sigma ellipse of a 2x2 covariance (rotated measured-CoP ellipses).

The semi-axes are the square roots of the covariance eigenvalues and the
angle is the major axis from +x, in ``(-90, 90]`` degrees.  The web mirror
(``groundReactionUncertainty.ts`` ``covarianceEllipse``) is held to the same
cases in ``groundReactionUncertainty.test.ts``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.plotting.renderers._cop_ellipse import (
    COVARIANCE,
    PER_AXIS,
    covariance_ellipse,
    marker_ellipse,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _rotated(
    major: float, minor: float, angle_deg: float
) -> tuple[float, float, float]:
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    cov = rot @ np.diag([major**2, minor**2]) @ rot.T
    return float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])


# (var_x, var_y, cov_xy) -> (semi-major, semi-minor, angle in degrees); the
# same table is in groundReactionUncertainty.test.ts.
CASES = [
    ((4e-4, 2.5e-5, 0.0), (0.02, 0.005, 0.0)),
    ((2.5e-5, 4e-4, 0.0), (0.02, 0.005, 90.0)),
    ((1e-4, 1e-4, 0.0), (0.01, 0.01, 0.0)),
    ((2.5e-4, 2.5e-4, 1.5e-4), (0.02, 0.01, 45.0)),
    ((2.5e-4, 2.5e-4, -1.5e-4), (0.02, 0.01, -45.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]


@pytest.mark.parametrize(("cov", "expected"), CASES)
def test_table_cases(cov, expected) -> None:
    assert covariance_ellipse(*cov) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("angle", [-89.0, -30.0, 0.0, 17.0, 60.0, 90.0])
def test_a_rotated_covariance_recovers_its_axes(angle: float) -> None:
    major, minor, got = covariance_ellipse(*_rotated(0.03, 0.01, angle))
    assert (major, minor) == pytest.approx((0.03, 0.01), rel=1e-9)
    assert got == pytest.approx(angle, abs=1e-9)


def test_projection_on_each_axis_is_that_axis_sigma() -> None:
    var_x, var_y, cov_xy = _rotated(0.03, 0.01, 35.0)
    major, minor, angle = covariance_ellipse(var_x, var_y, cov_xy)
    a = math.radians(angle)
    half_x = math.hypot(major * math.cos(a), minor * math.sin(a))
    half_y = math.hypot(major * math.sin(a), minor * math.cos(a))
    assert (half_x, half_y) == pytest.approx((math.sqrt(var_x), math.sqrt(var_y)))


@pytest.mark.parametrize(
    "bad",
    [
        (-1e-4, 1e-4, 0.0),
        (1e-4, math.nan, 0.0),
        (1e-4, 1e-4, math.inf),
        (1e-4, 1e-4, 2e-4),
    ],
)
def test_a_non_covariance_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="covariance"):
        covariance_ellipse(*bad)


def _traces(cov=None, sx=None, sy=None, worse=0.01):
    sigma = {"x": [sx], "y": [sy], "z": [worse]}
    if cov is None:
        return sigma, None
    return sigma, {"x": [cov[0]], "y": [cov[1]], "z": [cov[2]]}


def test_marker_prefers_the_covariance() -> None:
    sigma, cov = _traces(cov=_rotated(0.02, 0.005, 30.0), sx=0.02, sy=0.02)
    marker = marker_ellipse(sigma, cov, 0)
    assert marker[:3] == pytest.approx((0.02, 0.005, 30.0))
    assert marker.kind == COVARIANCE


def test_marker_falls_back_to_per_axis_then_worse_axis() -> None:
    assert marker_ellipse(*_traces(sx=0.02, sy=0.005), 0) == (
        0.02,
        0.005,
        0.0,
        PER_AXIS,
    )
    assert marker_ellipse(*_traces(), 0) == (0.01, 0.01, 0.0, PER_AXIS)
    sigma, cov = _traces(cov=(1e-4, None, 0.0), sx=0.02, sy=0.005)
    assert marker_ellipse(sigma, cov, 0) == (0.02, 0.005, 0.0, PER_AXIS)
